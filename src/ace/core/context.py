"""
Context data structure for ACE framework.

The Context acts as an evolving knowledge base that accumulates strategies,
insights, and domain-specific knowledge through incremental updates.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ContextBullet:
    """
    A single bullet point in the context.
    
    Attributes:
        bullet_id: Unique identifier (e.g., "calc-00001", "api-00042")
        content: The actual content/strategy/insight
        section: Which section this bullet belongs to
        helpful_count: Number of times this bullet was marked as helpful
        harmful_count: Number of times this bullet was marked as harmful
        neutral_count: Number of times this bullet was marked as neutral
        usage_count: Total number of times this bullet was used
        created_at: Timestamp when this bullet was created
    """
    bullet_id: str
    content: str
    section: str
    helpful_count: int = 0
    harmful_count: int = 0
    neutral_count: int = 0
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bullet_id": self.bullet_id,
            "content": self.content,
            "section": self.section,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "neutral_count": self.neutral_count,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextBullet":
        """Create from dictionary."""
        return cls(**data)
    
    def format_with_metadata(self) -> str:
        """Format bullet with metadata for display in context."""
        return f"[{self.bullet_id}] helpful={self.helpful_count} harmful={self.harmful_count} :: {self.content}"
    
    def update_tag(self, tag: str) -> None:
        """Update counts based on tag (helpful/harmful/neutral)."""
        if tag == "helpful":
            self.helpful_count += 1
        elif tag == "harmful":
            self.harmful_count += 1
        elif tag == "neutral":
            self.neutral_count += 1
        self.usage_count += 1


class Context:
    """
    Evolving context that accumulates knowledge through incremental updates.
    
    The Context implements the grow-and-refine principle from the ACE paper:
    - Only ADD operations (no EDIT/DELETE to prevent context collapse)
    - Structured sections for organization
    - Metadata tracking for each bullet
    """
    
    def __init__(
        self,
        sections: List[str],
        max_bullets_per_section: Optional[int] = None
    ):
        """
        Initialize a new Context.
        
        Args:
            sections: List of section names (e.g., ["strategies", "common_mistakes"])
            max_bullets_per_section: Optional maximum bullets per section (None = unlimited)
        """
        self.sections = sections
        self.max_bullets_per_section = max_bullets_per_section
        
        # Store bullets organized by section
        self.bullets: Dict[str, List[ContextBullet]] = {
            section: [] for section in sections
        }
        
        # Track bullet IDs for quick lookup
        self.bullet_index: Dict[str, ContextBullet] = {}
        
        # Counter for generating unique IDs per section
        self.section_counters: Dict[str, int] = {
            section: 0 for section in sections
        }
    
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None
    ) -> ContextBullet:
        """
        Add a new bullet to the context (incremental delta update).
        
        Args:
            section: Section to add the bullet to
            content: Content of the bullet
            bullet_id: Optional custom bullet ID (auto-generated if not provided)
            
        Returns:
            The created ContextBullet
            
        Raises:
            ValueError: If section doesn't exist or max bullets reached
        """
        if section not in self.sections:
            raise ValueError(f"Section '{section}' not found. Available: {self.sections}")
        
        if self.max_bullets_per_section is not None:
            if len(self.bullets[section]) >= self.max_bullets_per_section:
                raise ValueError(
                    f"Section '{section}' has reached max bullets ({self.max_bullets_per_section})"
                )
        
        # Generate bullet ID if not provided
        if bullet_id is None:
            # Create prefix from section name (e.g., "strategies" -> "strat")
            prefix = self._get_section_prefix(section)
            self.section_counters[section] += 1
            bullet_id = f"{prefix}-{self.section_counters[section]:05d}"
        
        # Create and add bullet
        bullet = ContextBullet(
            bullet_id=bullet_id,
            content=content,
            section=section
        )
        
        self.bullets[section].append(bullet)
        self.bullet_index[bullet_id] = bullet
        
        return bullet
    
    def get_bullet(self, bullet_id: str) -> Optional[ContextBullet]:
        """Get a bullet by its ID."""
        return self.bullet_index.get(bullet_id)
    
    def update_bullet_tags(self, bullet_tags: List[Dict[str, str]]) -> None:
        """
        Update bullet metadata based on tags from Reflector.
        
        Args:
            bullet_tags: List of dicts with 'id' and 'tag' keys
                        e.g., [{"id": "calc-00001", "tag": "helpful"}]
        """
        for item in bullet_tags:
            bullet_id = item.get("id")
            tag = item.get("tag")
            
            if bullet_id and tag:
                bullet = self.get_bullet(bullet_id)
                if bullet:
                    bullet.update_tag(tag)
    
    def format_for_prompt(self, include_metadata: bool = True) -> str:
        """
        Format the context for inclusion in prompts.
        
        Args:
            include_metadata: Whether to include helpful/harmful counts
            
        Returns:
            Formatted string representation of the context
        """
        lines = []
        
        for section in self.sections:
            if self.bullets[section]:
                # Section header
                lines.append(f"\n## {section.replace('_', ' ').title()}")
                lines.append("")
                
                # Bullets
                for bullet in self.bullets[section]:
                    if include_metadata:
                        lines.append(f"- {bullet.format_with_metadata()}")
                    else:
                        lines.append(f"- [{bullet.bullet_id}] {bullet.content}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the context."""
        total_bullets = sum(len(bullets) for bullets in self.bullets.values())
        
        section_stats = {}
        for section in self.sections:
            bullets = self.bullets[section]
            section_stats[section] = {
                "count": len(bullets),
                "helpful": sum(b.helpful_count for b in bullets),
                "harmful": sum(b.harmful_count for b in bullets),
                "neutral": sum(b.neutral_count for b in bullets),
            }
        
        return {
            "total_bullets": total_bullets,
            "sections": section_stats,
        }
    
    def to_json(self) -> str:
        """Serialize context to JSON string."""
        data = {
            "sections": self.sections,
            "max_bullets_per_section": self.max_bullets_per_section,
            "bullets": {
                section: [bullet.to_dict() for bullet in bullets]
                for section, bullets in self.bullets.items()
            },
            "section_counters": self.section_counters,
        }
        return json.dumps(data, indent=2)
    
    def save(self, filepath: str) -> None:
        """Save context to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Context":
        """Load context from JSON string."""
        data = json.loads(json_str)
        
        context = cls(
            sections=data["sections"],
            max_bullets_per_section=data.get("max_bullets_per_section"),
        )
        
        # Restore bullets
        for section, bullets_data in data["bullets"].items():
            for bullet_data in bullets_data:
                bullet = ContextBullet.from_dict(bullet_data)
                context.bullets[section].append(bullet)
                context.bullet_index[bullet.bullet_id] = bullet
        
        # Restore counters
        context.section_counters = data["section_counters"]
        
        return context
    
    @classmethod
    def load(cls, filepath: str) -> "Context":
        """Load context from a JSON file."""
        with open(filepath, "r") as f:
            return cls.from_json(f.read())
    
    def _get_section_prefix(self, section: str) -> str:
        """Generate a short prefix from section name for bullet IDs."""
        # Use first 5 chars of section name, remove underscores
        return section[:5].lower().replace("_", "")

