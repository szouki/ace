"""
Output schemas for ACE framework components.

These schemas define the expected JSON structure for Generator, Reflector, and Curator
outputs when using structured outputs with OpenAI.
"""

from typing import List, Dict, Any

# Generator output schema
GENERATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Chain of thought reasoning and detailed analysis"
        },
        "bullet_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of bullet IDs from the context that were helpful"
        },
        "final_answer": {
            "type": "string",
            "description": "The final answer or output for the task"
        }
    },
    "required": ["reasoning", "bullet_ids", "final_answer"],
    "additionalProperties": False
}

# Reflector output schema
REFLECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Analysis and reasoning process"
        },
        "error_identification": {
            "type": "string",
            "description": "What specifically went wrong in the reasoning"
        },
        "root_cause_analysis": {
            "type": "string",
            "description": "Why the error occurred and what concept was misunderstood"
        },
        "correct_approach": {
            "type": "string",
            "description": "What should have been done instead"
        },
        "key_insight": {
            "type": "string",
            "description": "Key strategy, formula, or principle to remember to avoid this error"
        },
        "bullet_tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Bullet ID"
                    },
                    "tag": {
                        "type": "string",
                        "enum": ["helpful", "harmful", "neutral"],
                        "description": "Tag indicating whether the bullet was helpful, harmful, or neutral"
                    }
                },
                "required": ["id", "tag"],
                "additionalProperties": False
            },
            "description": "Tags for each bullet that was used"
        }
    },
    "required": ["reasoning", "error_identification", "root_cause_analysis", "correct_approach", "key_insight", "bullet_tags"],
    "additionalProperties": False
}

# Curator output schema
CURATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Chain of thought and reasoning process for curation decisions"
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["ADD", "UPDATE", "DELETE"],
                        "description": "Type of operation to perform"
                    },
                    "section": {
                        "type": ["string", "null"],
                        "description": "Section name (required for ADD operations, null otherwise)"
                    },
                    "content": {
                        "type": ["string", "null"],
                        "description": "Content of the bullet (required for ADD and UPDATE operations, null otherwise)"
                    },
                    "bullet_id": {
                        "type": ["string", "null"],
                        "description": "Bullet ID (required for UPDATE and DELETE operations, null otherwise)"
                    }
                },
                "required": ["type", "section", "content", "bullet_id"],
                "additionalProperties": False
            },
            "description": "List of operations to perform on the context"
        }
    },
    "required": ["reasoning", "operations"],
    "additionalProperties": False
}


def get_schema(component: str) -> Dict[str, Any]:
    """
    Get the output schema for a specific component.
    
    Args:
        component: Name of the component ('generator', 'reflector', or 'curator')
        
    Returns:
        JSON schema dictionary for the component's output
        
    Raises:
        ValueError: If component name is not recognized
    """
    schemas = {
        "generator": GENERATOR_SCHEMA,
        "reflector": REFLECTOR_SCHEMA,
        "curator": CURATOR_SCHEMA
    }
    
    component_lower = component.lower()
    if component_lower not in schemas:
        raise ValueError(f"Unknown component: {component}. Must be one of: {', '.join(schemas.keys())}")
    
    return schemas[component_lower]

