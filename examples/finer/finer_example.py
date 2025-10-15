"""
FINER Example: Financial NER with ACE Framework

This example demonstrates how to use ACE for financial Named Entity Recognition tasks
using the FINER-139 dataset from Hugging Face.
It shows offline adaptation - building a context from a training set.
"""

import os
import sys
import random
from dotenv import load_dotenv
from datasets import load_dataset

# Add project root directory to path to import ace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ace import Context, Generator, Reflector, Curator
from src.ace.llm import OpenAIClient
from src.ace.utils import PromptManager
from finer_helper import (
    format_ner_example,
    extract_entities,
    calculate_metrics,
    parse_predicted_tags
)



"""
Run FINER example demonstrating ACE framework.
"""
# Load environment variables
load_dotenv()

# Config
TRAIN_N = int(os.getenv("TRAIN_N", 50))  # Number of training examples
SEED = int(os.getenv("SEED", 42))
random.seed(SEED)

print("=" * 80)
print("FINER Example: Financial NER with ACE Framework")
print("=" * 80)
print()

# Initialize LLM client
print("Initializing GPT-5 client...")
llm = OpenAIClient(
    model="gpt-5-mini",
    # temperature=0.7,
)
print(f"✓ LLM client initialized: {llm}")
print()

# Load FINER-139 dataset
print("Loading FINER-139 dataset from Hugging Face...")
dataset = load_dataset("nlpaueb/finer-139")

# Get label names - this is how you access class names from integer IDs
label_feature = dataset["train"].features["ner_tags"].feature
label_names = label_feature.names  # List where index = integer class ID

# Create integer to class name mapping
label_id_to_name = {i: label_names[i] for i in range(len(label_names))}

# Extract unique entity types (without B- and I- prefixes)
entity_types = sorted(set(l[2:] for l in label_names if l != 'O'))

print(f"✓ Dataset loaded")
print(f"  - Train size: {len(dataset['train'])}")
print(f"  - Validation size: {len(dataset['validation'])}")
print(f"  - Test size: {len(dataset['test'])}")
print(f"  - Number of entity types: {len(entity_types)}")
print(f"  - Total labels (with IOB2): {len(label_names)}")
print(f"  - Example label mappings:")
print(f"    0 -> '{label_names[0]}'")
print(f"    1 -> '{label_names[1]}'")
print(f"    2 -> '{label_names[2]}'")
print()

# Sample training examples
print(f"Sampling {TRAIN_N} training examples...")
train_indices = list(range(len(dataset["train"])))
random.shuffle(train_indices)
train_indices = train_indices[:TRAIN_N]
train_examples = [dataset["train"][i] for i in train_indices]
print(f"✓ Sampled {len(train_examples)} examples")
print()

# Load FINER prompts (if available)
print("Loading FINER prompts...")
try:
    prompt_manager = PromptManager(prompts_file="examples/finer/prompts_finer.yml")
    finer_prompts = prompt_manager.load_prompts()
    print(f"✓ Loaded prompts for: {', '.join(finer_prompts.keys())}")
except FileNotFoundError:
    print("⚠ No FINER prompts file found, using default prompts")
    finer_prompts = {}
print()

# Build comprehensive dataset information for the LLM
# Create the label mapping string
label_mapping_str = "\n".join([f"{i} -> '{label_names[i]}'" for i in range(len(label_names))])

# Format dataset info from template
dataset_info_template = finer_prompts.get("dataset_info", "")
dataset_info = dataset_info_template.format(
    num_labels=len(label_names),
    max_label_id=len(label_names)-1,
    label_mapping=label_mapping_str
) if dataset_info_template else ""

# Initialize Context with FINER-specific sections
print("Creating context with financial NER sections...")
context = Context(
    sections=[
        "dataset_information",
        "entity_types_and_definitions",
        "extraction_strategies",
        "common_mistakes",
    ]
)

# Add dataset information as initial context
context.add_bullet(
    section="dataset_information",
    content=dataset_info.strip()
)

print(f"✓ Context created with {len(context.sections)} sections")
print(f"✓ Added comprehensive dataset information with all {len(entity_types)} entity types")
print()

# Initialize ACE components
print("Initializing ACE components...")
generator = Generator(
    llm=llm,
    context=context,
    prompt_template=finer_prompts.get("generator")
)

reflector = Reflector(
    llm=llm,
    prompt_template=finer_prompts.get("reflector")
)

curator = Curator(
    llm=llm,
    prompt_template=finer_prompts.get("curator")
)
print("✓ All ACE components initialized")
print()

# Training loop: Process examples and evolve context
print("=" * 80)
print("Training: Building Context from Examples")
print("=" * 80)
print()

num_train_steps = min(3, len(train_examples))  # Process first 5 examples

# Track metrics across all examples
all_metrics = []

for step, example in enumerate(train_examples[:num_train_steps], 1):
    print(f"\n{'='*80}")
    print(f"Training Step {step}/{num_train_steps}")
    print(f"{'='*80}\n")
    
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]
    
    # Format the task
    text = " ".join(tokens)
    
    # Show example with integer tags
    sample_tokens_str = str(tokens[:15]) if len(tokens) > 15 else str(tokens)
    sample_tokens_display = f"{sample_tokens_str}{'...' if len(tokens) > 15 else ''}"
    
    # Format task from template
    task_template = finer_prompts.get("ner_task", "")
    task = task_template.format(
        sample_tokens=sample_tokens_display,
        text=text
    ) if task_template else ""
    
    # Get ground truth
    ground_truth_entities = extract_entities(tokens, ner_tags, label_names)
    ground_truth_text = "\n".join([
        f"{e['text']}: {e['label']}" for e in ground_truth_entities
    ])
    
    # Ground truth with integer tags
    ground_truth = f"""
Ground Truth Format:
- tokens: {tokens}
- ner_tags (FULL): {ner_tags}
- Total tokens: {len(tokens)}

Entities extracted:
{ground_truth_text}
"""
    
    print(f"Text: {text[:100]}...")
    print(f"\nGround truth entities: {len(ground_truth_entities)}")
    for ent in ground_truth_entities:  # Show ALL entities
        print(f"  - {ent['text']}: {ent['label']}")
    print(f"\nGround truth ner_tags (FULL): {ner_tags}")
    print(f"Total tokens: {len(tokens)}")
    print()
    
    # Step 1: Generate
    print("-" * 80)
    print("Generating extraction...")
    print("-" * 80)
    
    try:
        result = generator.generate(
            prompt_variables={
                "task_input": task,
                "reflection": ""
            }
        )
        
        print(f"\nBullets used: {result.get('bullet_ids', [])}")
        print(f"\n{'='*80}")
        print("FULL MODEL OUTPUT:")
        print(f"{'='*80}")
        print(result['final_answer'])
        print(f"{'='*80}")
        
        # Parse predicted tags and calculate metrics
        predicted_tags = parse_predicted_tags(result['final_answer'], len(tokens))
        
        if predicted_tags:
            print(f"\nPredicted tags (FULL): {predicted_tags}")
            print(f"Predicted length: {len(predicted_tags)}")
            print(f"Ground truth length: {len(ner_tags)}")
            
            metrics = calculate_metrics(predicted_tags, ner_tags)
            all_metrics.append(metrics)
            
            print("\n" + "=" * 60)
            print("METRICS (comparing predicted vs ground truth tags)")
            print("=" * 60)
            print(f"Accuracy:           {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']} tokens)")
            print(f"Precision (entities): {metrics['precision']:.2%} (TP={metrics['true_positives']}, FP={metrics['false_positives']})")
            print(f"Recall (entities):    {metrics['recall']:.2%} (TP={metrics['true_positives']}, FN={metrics['false_negatives']})")
            print(f"F1 Score:           {metrics['f1']:.2%}")
            print(f"Predicted entities: {metrics['predicted_entities']} (non-zero tags)")
            print(f"Ground truth entities: {metrics['ground_truth_entities']} (non-zero tags)")
            print("=" * 60)
        else:
            print("\n⚠ Could not parse predicted tags for metrics calculation")
            all_metrics.append(None)
        
        print()
        
        # Step 2: Reflect
        print("-" * 80)
        print("Reflecting on extraction...")
        print("-" * 80)
        
        reflection = reflector.reflect(
            context=context,
            used_bullet_ids=result.get('bullet_ids', []),
            prompt_variables={
                "task_input": task,
                "model_reasoning": result['reasoning'],
                "model_output": result['final_answer'],
                "ground_truth": ground_truth,
                "environment_feedback": ""
            }
        )
        
        print(f"\nKey Insight: {reflection['key_insight'][:150]}...")
        print()
        
        # Step 3: Curate
        print("-" * 80)
        print("Curating new insights...")
        print("-" * 80)
        
        curation = curator.curate(
            context=context,
            prompt_variables={
                "reflection": reflection,
                "task_input": f"Financial Numeric Entity Recognition - Training example {step}/{num_train_steps}",
                "current_step": step,
                "total_steps": num_train_steps
            }
        )
        
        print(f"\nOperations: {len(curation['operations'])} new insights added")
        for i, op in enumerate(curation['operations'][:2], 1):  # Show first 2
            print(f"  {i}. [{op['section']}] {op['content'][:80]}...")
        if len(curation['operations']) > 2:
            print(f"  ... and {len(curation['operations']) - 2} more")
        print()
        
    except Exception as e:
        print(f"⚠ Error processing example: {e}")
        continue

# Show average metrics across all examples
print("\n" + "=" * 80)
print("Training Metrics Summary")
print("=" * 80)
print()

valid_metrics = [m for m in all_metrics if m is not None]
if valid_metrics:
    avg_accuracy = sum(m['accuracy'] for m in valid_metrics) / len(valid_metrics)
    avg_precision = sum(m['precision'] for m in valid_metrics) / len(valid_metrics)
    avg_recall = sum(m['recall'] for m in valid_metrics) / len(valid_metrics)
    avg_f1 = sum(m['f1'] for m in valid_metrics) / len(valid_metrics)
    total_tp = sum(m['true_positives'] for m in valid_metrics)
    total_fp = sum(m['false_positives'] for m in valid_metrics)
    total_fn = sum(m['false_negatives'] for m in valid_metrics)
    
    print(f"Average Metrics (across {len(valid_metrics)} examples):")
    print(f"  Accuracy:   {avg_accuracy:.2%}")
    print(f"  Precision:  {avg_precision:.2%}")
    print(f"  Recall:     {avg_recall:.2%}")
    print(f"  F1 Score:   {avg_f1:.2%}")
    print()
    print(f"Total Counts:")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
else:
    print("⚠ No valid metrics to summarize")
print()

# Show final context stats
print("=" * 80)
print("Final Context Statistics")
print("=" * 80)
print()

stats = context.get_stats()
print(f"Total bullets: {stats['total_bullets']}")
for section, section_stats in stats['sections'].items():
    print(f"  {section}: {section_stats['count']} bullets")
print()

# Save context
context_file = "finer_context.json"
context.save(context_file)
print(f"✓ Context saved to {context_file}")
print()

# Test on a new example
print("=" * 80)
print("Testing on New Example")
print("=" * 80)
print()

test_example = dataset["test"][0]
test_tokens = test_example["tokens"]
test_text = " ".join(test_tokens)

# Show sample tokens
sample_test_tokens = str(test_tokens[:15]) if len(test_tokens) > 15 else str(test_tokens)
sample_test_tokens_display = f"{sample_test_tokens}{'...' if len(test_tokens) > 15 else ''}"

# Format test task from template
test_task_template = finer_prompts.get("ner_task", "")
test_task = test_task_template.format(
    sample_tokens=sample_test_tokens_display,
    text=test_text
) if test_task_template else ""

print(f"Test text: {test_text[:150]}...")
print()

result = generator.generate(
    prompt_variables={
        "task_input": test_task,
        "reflection": ""
    }
)

print("Generated extraction (FULL):")
print(result['final_answer'])
print()
print(f"Bullets used: {result.get('bullet_ids', [])}")
print()

# Ground truth
test_entities = extract_entities(test_tokens, test_example["ner_tags"], label_names)
print(f"\nGround truth entities ({len(test_entities)}) - ALL:")
for ent in test_entities:
    print(f"  - {ent['text']}: {ent['label']}")
print(f"\nGround truth ner_tags (FULL): {test_example['ner_tags']}")
print(f"Total tokens: {len(test_tokens)}")
print()

print("=" * 80)
print("FINER Example Complete!")
print("=" * 80)
print()
print("Summary:")
print(f"- Processed {num_train_steps} training examples")
print(f"- Built context with {stats['total_bullets']} insights")
print(f"- Context can now be reused for financial NER tasks!")
print()

