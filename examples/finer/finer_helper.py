"""
FINER Helper Functions

Utility functions for financial Named Entity Recognition tasks.
"""

import json
import re


def format_ner_example(tokens, ner_tags, label_names):
    """Format a NER example as a readable string."""
    lines = []
    lines.append("Text (with tokens):")
    lines.append(" ".join(tokens))
    lines.append("\nLabeled entities:")
    
    current_entity = []
    current_label = None
    
    for token, tag_id in zip(tokens, ner_tags):
        label = label_names[tag_id]
        
        if label.startswith("B-"):
            if current_entity:
                lines.append(f"  {' '.join(current_entity)}: {current_label}")
            current_entity = [token]
            current_label = label[2:]
        elif label.startswith("I-") and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                lines.append(f"  {' '.join(current_entity)}: {current_label}")
                current_entity = []
                current_label = None
    
    if current_entity:
        lines.append(f"  {' '.join(current_entity)}: {current_label}")
    
    return "\n".join(lines)


def extract_entities(tokens, ner_tags, label_names):
    """Extract entities from NER tags."""
    entities = []
    current_entity = []
    current_label = None
    current_start = None
    
    for i, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
        label = label_names[tag_id]
        
        if label.startswith("B-"):
            if current_entity:
                entities.append({
                    "text": " ".join(current_entity),
                    "label": current_label,
                    "start": current_start,
                    "end": i - 1
                })
            current_entity = [token]
            current_label = label[2:]
            current_start = i
        elif label.startswith("I-") and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append({
                    "text": " ".join(current_entity),
                    "label": current_label,
                    "start": current_start,
                    "end": i - 1
                })
                current_entity = []
                current_label = None
    
    if current_entity:
        entities.append({
            "text": " ".join(current_entity),
            "label": current_label,
            "start": current_start,
            "end": len(tokens) - 1
        })
    
    return entities


def calculate_metrics(predicted_tags, ground_truth_tags):
    """
    Calculate metrics comparing predicted and ground truth NER tags.
    
    Args:
        predicted_tags: List of predicted integer tags
        ground_truth_tags: List of ground truth integer tags
        
    Returns:
        Dictionary with accuracy and precision metrics
    """
    if len(predicted_tags) != len(ground_truth_tags):
        # If lengths don't match, pad or truncate
        min_len = min(len(predicted_tags), len(ground_truth_tags))
        predicted_tags = predicted_tags[:min_len]
        ground_truth_tags = ground_truth_tags[:min_len]
    
    # Overall accuracy
    correct = sum(1 for p, g in zip(predicted_tags, ground_truth_tags) if p == g)
    total = len(ground_truth_tags)
    accuracy = correct / total if total > 0 else 0.0
    
    # Precision for non-zero labels (entity labels, not 'O')
    # True Positives: predicted non-zero and matches ground truth
    # False Positives: predicted non-zero but doesn't match ground truth
    predicted_entities = [(i, p) for i, p in enumerate(predicted_tags) if p != 0]
    
    if len(predicted_entities) == 0:
        precision = 0.0
        tp = 0
        fp = 0
    else:
        tp = sum(1 for i, p in predicted_entities if ground_truth_tags[i] == p)
        fp = len(predicted_entities) - tp
        precision = tp / len(predicted_entities)
    
    # Recall for non-zero labels
    ground_truth_entities = [(i, g) for i, g in enumerate(ground_truth_tags) if g != 0]
    
    if len(ground_truth_entities) == 0:
        recall = 0.0
        fn = 0
    else:
        fn = sum(1 for i, g in ground_truth_entities if predicted_tags[i] != g)
        recall = tp / len(ground_truth_entities) if len(ground_truth_entities) > 0 else 0.0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "predicted_entities": len(predicted_entities),
        "ground_truth_entities": len(ground_truth_entities)
    }


def parse_predicted_tags(model_output, num_tokens):
    """
    Try to parse predicted NER tags from model output.
    
    Args:
        model_output: The model's output string or list
        num_tokens: Expected number of tokens
        
    Returns:
        List of integers (predicted tags), or None if parsing fails
    """
    # If it's already a list, check if it's valid
    if isinstance(model_output, list):
        if all(isinstance(x, int) for x in model_output):
            return model_output
        # Try to convert to integers
        try:
            return [int(x) for x in model_output]
        except:
            pass
    
    # If it's a string, try to parse it
    if isinstance(model_output, str):
        # Try to extract JSON object
        try:
            json_match = re.search(r'\{.*\}', model_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'ner_tags' in data:
                    tags = data['ner_tags']
                    if isinstance(tags, list):
                        return tags
                if 'final_answer' in data:
                    answer = data['final_answer']
                    if isinstance(answer, list):
                        return answer
        except:
            pass
        
        # Try to find a list directly in the string
        try:
            list_match = re.search(r'\[[\d\s,]+\]', model_output)
            if list_match:
                list_str = list_match.group()
                tags = json.loads(list_str)
                if isinstance(tags, list) and all(isinstance(x, int) for x in tags):
                    return tags
        except:
            pass
    
    return None

