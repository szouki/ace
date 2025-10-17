"""
Context data structure for ACE framework.

The Context acts as an evolving knowledge base that accumulates strategies,
insights, and domain-specific knowledge through incremental updates.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np

if TYPE_CHECKING:
    from ace.embeddings.base import BaseEmbedder


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
        embedding: Optional embedding vector for semantic similarity
    """
    bullet_id: str
    content: str
    section: str
    helpful_count: int = 0
    harmful_count: int = 0
    neutral_count: int = 0
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "bullet_id": self.bullet_id,
            "content": self.content,
            "section": self.section,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "neutral_count": self.neutral_count,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
        }
        # Store embedding as list for JSON serialization
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextBullet":
        """Create from dictionary."""
        # Convert embedding list back to numpy array if present
        data_copy = data.copy()
        if "embedding" in data_copy and data_copy["embedding"] is not None:
            data_copy["embedding"] = np.array(data_copy["embedding"], dtype=np.float32)
        return cls(**data_copy)
    
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
    - Incremental delta updates (ADD operations)
    - Semantic de-duplication using embeddings
    - Grow-and-refine to maintain context quality
    - Structured sections for organization
    - Metadata tracking for each bullet
    """
    
    def __init__(
        self,
        sections: List[str],
        max_bullets: Optional[int] = None,
        embedder: Optional["BaseEmbedder"] = None,
        similarity_threshold: float = 0.85,
        refinement_mode: str = "lazy"
    ):
        """
        Initialize a new Context.
        
        Args:
            sections: List of section names (e.g., ["strategies", "common_mistakes"])
            max_bullets: Optional maximum total number of bullets (None = unlimited)
            embedder: Optional embedder for semantic similarity (required for de-duplication)
            similarity_threshold: Threshold for considering bullets as duplicates (0.0-1.0)
            refinement_mode: "proactive" (refine after each delta) or "lazy" (refine only when max exceeded)
        """
        self.sections = sections
        self.max_bullets = max_bullets
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.refinement_mode = refinement_mode
        
        if refinement_mode not in ["proactive", "lazy"]:
            raise ValueError(f"refinement_mode must be 'proactive' or 'lazy', got: {refinement_mode}")
        
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
        bullet_id: Optional[str] = None,
        skip_refinement: bool = False
    ) -> ContextBullet:
        """
        Add a new bullet to the context (incremental delta update).
        
        Args:
            section: Section to add the bullet to
            content: Content of the bullet
            bullet_id: Optional custom bullet ID (auto-generated if not provided)
            skip_refinement: If True, skip automatic refinement (useful for batch adds)
            
        Returns:
            The created ContextBullet
            
        Raises:
            ValueError: If section doesn't exist
        """
        if section not in self.sections:
            raise ValueError(f"Section '{section}' not found. Available: {self.sections}")
        
        # Generate bullet ID if not provided
        if bullet_id is None:
            # Create prefix from section name (e.g., "strategies" -> "strat")
            prefix = self._get_section_prefix(section)
            self.section_counters[section] += 1
            bullet_id = f"{prefix}-{self.section_counters[section]:05d}"
        
        # Generate embedding if embedder is available
        embedding = None
        if self.embedder is not None:
            embedding = self.embedder.embed(content)
        
        # Create and add bullet
        bullet = ContextBullet(
            bullet_id=bullet_id,
            content=content,
            section=section,
            embedding=embedding
        )
        
        self.bullets[section].append(bullet)
        self.bullet_index[bullet_id] = bullet
        
        # Apply grow-and-refine based on mode
        if not skip_refinement:
            if self.refinement_mode == "proactive":
                # Refine after each addition
                self.refine()
            elif self.refinement_mode == "lazy" and self.max_bullets is not None:
                # Refine only if we exceeded max bullets
                total_bullets = sum(len(bullets) for bullets in self.bullets.values())
                if total_bullets > self.max_bullets:
                    self.refine()
        
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
    
    def deduplicate(self) -> int:
        """
        De-duplicate bullets based on semantic similarity.
        
        Uses vectorized operations to efficiently compute pairwise similarities
        and merge semantically similar bullets. This is significantly faster than
        nested loops for large contexts.
        
        When duplicates are found, keeps the one with higher value (more helpful tags)
        and merges their metadata.
        
        Returns:
            Number of bullets removed during de-duplication
            
        Raises:
            ValueError: If embedder is not configured
        """
        if self.embedder is None:
            raise ValueError("Embedder must be configured to perform de-duplication")
        
        removed_count = 0
        
        # Process each section independently
        for section in self.sections:
            bullets = self.bullets[section]
            
            # Skip if section is empty or has only one bullet
            if len(bullets) <= 1:
                continue
            
            # Filter bullets with embeddings
            bullets_with_embeddings = [
                (i, b) for i, b in enumerate(bullets) 
                if b.embedding is not None
            ]
            
            if len(bullets_with_embeddings) <= 1:
                continue
            
            # Extract embeddings and create value scores
            indices = [i for i, _ in bullets_with_embeddings]
            embeddings_matrix = np.array([b.embedding for _, b in bullets_with_embeddings])
            values = np.array([
                b.helpful_count - b.harmful_count 
                for _, b in bullets_with_embeddings
            ])
            
            # Compute pairwise cosine similarities using vectorized operations
            # For normalized embeddings: similarity = dot product
            similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
            
            # Find pairs above threshold (upper triangle only to avoid duplicates)
            # Set diagonal to 0 to avoid self-comparison
            np.fill_diagonal(similarity_matrix, 0)
            duplicate_pairs = np.argwhere(similarity_matrix >= self.similarity_threshold)
            
            # Filter to upper triangle only (i < j)
            duplicate_pairs = duplicate_pairs[duplicate_pairs[:, 0] < duplicate_pairs[:, 1]]
            
            # Determine which bullets to remove based on value
            to_remove = set()
            for i, j in duplicate_pairs:
                print(f"Duplicate pair: {i}, {j}")
                # Skip if already marked for removal
                if i in to_remove or j in to_remove:
                    continue
                
                # Get actual bullets
                bullet_i = bullets_with_embeddings[i][1]
                bullet_j = bullets_with_embeddings[j][1]
                
                # Keep the higher value bullet, merge the other into it
                if values[j] > values[i]:
                    # Keep j, merge i into it
                    bullet_j.helpful_count += bullet_i.helpful_count
                    bullet_j.harmful_count += bullet_i.harmful_count
                    bullet_j.neutral_count += bullet_i.neutral_count
                    bullet_j.usage_count += bullet_i.usage_count
                    to_remove.add(i)
                else:
                    # Keep i, merge j into it
                    bullet_i.helpful_count += bullet_j.helpful_count
                    bullet_i.harmful_count += bullet_j.harmful_count
                    bullet_i.neutral_count += bullet_j.neutral_count
                    bullet_i.usage_count += bullet_j.usage_count
                    to_remove.add(j)
            
            # Remove duplicates (convert back to original indices)
            if to_remove:
                original_indices_to_remove = sorted(
                    [indices[i] for i in to_remove], 
                    reverse=True
                )
                for idx in original_indices_to_remove:
                    removed_bullet = bullets.pop(idx)
                    del self.bullet_index[removed_bullet.bullet_id]
                    removed_count += 1
        
        return removed_count
    
    def refine(self) -> Dict[str, int]:
        """
        Grow-and-refine: De-duplicate and optionally prune low-value bullets.
        
        This implements the grow-and-refine principle from the ACE paper:
        1. De-duplicate using semantic embeddings
        2. If still over max_bullets, prune lowest-value bullets
        
        Returns:
            Dictionary with refinement statistics:
                - deduplicated: Number of bullets removed via de-duplication
                - pruned: Number of bullets pruned to meet max_bullets
                - total_removed: Total bullets removed
        """
        stats = {
            "deduplicated": 0,
            "pruned": 0,
            "total_removed": 0
        }
        
        # Step 1: De-duplicate (if embedder is available)
        if self.embedder is not None:
            deduplicated = self.deduplicate()
            stats["deduplicated"] = deduplicated
            stats["total_removed"] += deduplicated
        
        # Step 2: Prune if still over max_bullets
        if self.max_bullets is not None:
            total_bullets = sum(len(bullets) for bullets in self.bullets.values())
            
            if total_bullets > self.max_bullets:
                bullets_to_remove = total_bullets - self.max_bullets
                
                # Collect all bullets with their value scores
                all_bullets_with_scores = []
                for section in self.sections:
                    for bullet in self.bullets[section]:
                        # Value = helpful - harmful, with usage_count as tiebreaker
                        value = bullet.helpful_count - bullet.harmful_count
                        all_bullets_with_scores.append((bullet, value, section))
                
                # Sort by value (ascending) so lowest value bullets come first
                all_bullets_with_scores.sort(key=lambda x: (x[1], x[0].usage_count))
                
                # Remove lowest-value bullets
                for i in range(bullets_to_remove):
                    bullet, value, section = all_bullets_with_scores[i]
                    self.bullets[section].remove(bullet)
                    del self.bullet_index[bullet.bullet_id]
                    stats["pruned"] += 1
                    stats["total_removed"] += 1
        
        return stats
    
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
            "max_bullets": self.max_bullets,
            "similarity_threshold": self.similarity_threshold,
            "refinement_mode": self.refinement_mode,
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
    def from_json(cls, json_str: str, embedder: Optional["BaseEmbedder"] = None) -> "Context":
        """
        Load context from JSON string.
        
        Args:
            json_str: JSON string to load from
            embedder: Optional embedder to attach to the context
        
        Returns:
            Loaded Context instance
        """
        data = json.loads(json_str)
        
        context = cls(
            sections=data["sections"],
            max_bullets=data.get("max_bullets"),
            embedder=embedder,
            similarity_threshold=data.get("similarity_threshold", 0.85),
            refinement_mode=data.get("refinement_mode", "lazy"),
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
    def load(cls, filepath: str, embedder: Optional["BaseEmbedder"] = None) -> "Context":
        """
        Load context from a JSON file.
        
        Args:
            filepath: Path to JSON file
            embedder: Optional embedder to attach to the context
            
        Returns:
            Loaded Context instance
        """
        with open(filepath, "r") as f:
            return cls.from_json(f.read(), embedder=embedder)
    
    def _get_section_prefix(self, section: str) -> str:
        """Generate a short prefix from section name for bullet IDs."""
        # Use first 5 chars of section name, remove underscores
        return section[:5].lower().replace("_", "")

