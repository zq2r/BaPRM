import torch
from internvl.conversation import get_conv_template

def _resolve_optional_bool(value, default):
    """
    Resolve an optional bool override.

    value:
        None / "auto": use default
        True / "true" / "1" / "yes": force True
        False / "false" / "0" / "no": force False
    """
    if value is None:
        return bool(default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("auto", "none", ""):
            return bool(default)
        if value in ("true", "1", "yes", "y"):
            return True
        if value in ("false", "0", "no", "n"):
            return False

    raise ValueError(f"Cannot parse optional bool value: {value}")

@torch.no_grad()
def batch_prm_weighted_mu(
    model,
    tokenizer,
    pixel_values,
    questions,
    num_patches_list,
    verbose=False,
    belief_use_conservatism=None,
    belief_conservatism_beta=None,
    belief_hybrid_lambda=None,
):
    """
    Basic BayesianPRM evaluator.

    For each <prm> marker:
        mu_heads = sigmoid(ensemble_prm_head(h_prm))      # [P, E]
        weights  = softmax(belief_head(h_prm, mu_heads)) # [P, E]
        mu_bayes = sum_m weights_m * mu_heads_m          # [P]

    This version does not compute uncertainty.
    """
    prm_token_id = tokenizer.convert_tokens_to_ids("<prm>")
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_context_token_id

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f"dynamic ViT batch size: {image_bs}")

    queries = []

    for idx, num_patches in enumerate(num_patches_list):
        question = questions[idx]

        if pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        template = get_conv_template(model.template)
        template.append_message(template.roles[0], "<image>\n" + question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * model.num_image_token * num_patches
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)
        queries.append(query)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
    tokenizer.padding_side = old_padding_side

    device = pixel_values.device
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    outputs = model.forward(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=torch.tensor(
            [1] * pixel_values.shape[0],
            dtype=torch.long,
            device=device,
        ),
        output_hidden_states=True,
        return_dict=True,
    )

    placeholder_mask = input_ids == prm_token_id

    if not placeholder_mask.any():
        return input_ids.new_zeros((0,), dtype=torch.float32)

    if not hasattr(model, "ensemble_prm_head") or model.ensemble_prm_head is None:
        raise RuntimeError(
            "BayesianPRM evaluation requires model.ensemble_prm_head."
        )

    if not hasattr(model, "belief_head") or model.belief_head is None:
        raise RuntimeError(
            "BayesianPRM evaluation requires model.belief_head. "
            "Please evaluate a BayesianPRM checkpoint, not an ensemble-only checkpoint."
        )

    hidden = outputs.hidden_states[-1]
    prm_h = hidden[placeholder_mask]  # [P, H]

    # ensemble_logits: [E, P]
    ensemble_logits = model.ensemble_prm_head(prm_h)
    ensemble_probs = torch.sigmoid(ensemble_logits.float()).clamp(
        1e-6,
        1.0 - 1e-6,
    )

    # mu_heads: [P, E]
    mu_heads = ensemble_probs.transpose(0, 1).contiguous()

    # Reliability posterior: alpha_rel = q_phi(z=m | c_t).
    # Shape: [P, E]
    belief_logits = model.belief_head(
        prm_h,
        mu_heads,
    )
    rel_weights = torch.softmax(
        belief_logits.float(),
        dim=-1,
    )

    # Conservative posterior and final hybrid posterior.
    #
    # These attributes are loaded from checkpoint config by InternVLChatModel.
    # If conservatism is disabled, this exactly reduces to the original
    # BayesianPRM evaluator.
    default_use_conservatism = bool(
        getattr(model, "belief_use_conservatism", False)
    )
    use_conservatism = _resolve_optional_bool(
        belief_use_conservatism,
        default_use_conservatism,
    )

    if belief_hybrid_lambda is None:
        hybrid_lambda = float(
            getattr(model, "belief_hybrid_lambda", 1.0)
        )
    else:
        hybrid_lambda = float(belief_hybrid_lambda)

    if belief_conservatism_beta is None:
        conservatism_beta = float(
            getattr(model, "belief_conservatism_beta", 0.1)
        )
    else:
        conservatism_beta = float(belief_conservatism_beta)

    if use_conservatism and hybrid_lambda < 1.0:
        temperature = max(
            conservatism_beta,
            1e-6,
        )

        # Conservative posterior:
        # alpha_con = softmax(-reward / beta_2).
        #
        # Here reward is represented by each ensemble head's predicted
        # correctness probability mu_m(c_t).
        con_weights = torch.softmax(
            -mu_heads / temperature,
            dim=-1,
        )

        # Hybrid posterior:
        # alpha_post = lambda * alpha_rel + (1-lambda) * alpha_con.
        post_weights = (
            hybrid_lambda * rel_weights
            + (1.0 - hybrid_lambda) * con_weights
        )

        # Numerical safety.
        post_weights = post_weights / post_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
    else:
        post_weights = rel_weights

    # Final BayesianPRM score.
    # Shape: [P]
    mu_bayes = (post_weights * mu_heads).sum(dim=-1)
    return mu_bayes