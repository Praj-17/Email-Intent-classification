from loguru import logger
import torch

def perform_email_prediction(email_text: str, model, tokenizer, device, all_labels: list[str], pred_threshold: float):
    """Performs intent prediction for a given email text."""
    logger.info(f"Performing prediction for email: {email_text[:50]}...")
    try:
        enc = tokenizer(email_text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
        
        logger.debug(f"Raw logits: {logits.tolist()}")

        probabilities_all = torch.sigmoid(logits).squeeze()
        logger.debug(f"Probabilities (post-sigmoid): {probabilities_all.tolist()}")
        
        predicted_multi_hot = (probabilities_all > pred_threshold).cpu().numpy().astype(int)
        logger.debug(f"Predicted multi-hot (threshold {pred_threshold}): {predicted_multi_hot.tolist()}")
        
        predicted_intents = [all_labels[i] for i, is_present in enumerate(predicted_multi_hot) if is_present == 1]
        
        logger.info(f"Prediction successful. Intents: {predicted_intents}")
        return {
            "email_text": email_text,
            "predicted_intents": predicted_intents,
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        # Return a structured error or raise HTTPException for FastAPI to handle
        return {
            "email_text": email_text,
            "predicted_intents": ["error during prediction"],
            "error_detail": str(e) # Optional: provide more error context
        }