import pandas as pd
import openai
from typing import List, Dict, Optional
import json
import time
import random
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalMisinfoDetectionDatasetGenerator:
    """
    Generate synthetic medical misinformation for detection model training.
    
    IMPORTANT: This tool is designed for academic research and AI safety.
    Generated content should never be used to spread actual misinformation.
    """
    
    def __init__(self, api_key: str, model_name: str = "QwQ", base_url: str = "http://localhost:8000/v1"):
        """
        Initialize the generator with your vLLM-served QwQ model
        """
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.generation_count = 0
        
        # Add research ethics disclaimer
        logger.warning("="*80)
        logger.warning("RESEARCH ETHICS NOTICE:")
        logger.warning("This tool generates synthetic misinformation for AI safety research.")
        logger.warning("Generated content must not be used to spread actual misinformation.")
        logger.warning("Ensure proper safeguards and ethical review before use.")
        logger.warning("="*80)
        
    def _test_connection(self) -> bool:
        """Test connection to the vLLM server"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello, please respond with 'OK'"}],
                max_tokens=10,
                temperature=0.1
            )
            return "OK" in response.choices[0].message.content
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
        
    def generate_false_article(self, original_article: str, title: str, difficulty: str) -> Optional[Dict]:
        """
        Generate a false version of a medical article at specified difficulty level
        """
        
        # Validate inputs
        if not original_article or not title or difficulty not in ["easy", "medium", "hard"]:
            logger.error(f"Invalid inputs: article length={len(original_article)}, title={bool(title)}, difficulty={difficulty}")
            return None
            
        # Define difficulty-specific prompts with better structure
        difficulty_prompts = {
            "easy": {
                "instruction": """Create a clearly false version of this medical article for misinformation detection training. The false version should contain obvious misinformation that would be easily spotted, including:
                - Completely unsupported miracle cures
                - Obviously exaggerated statistics (like 100% cure rates)
                - Claims that contradict basic medical knowledge
                - Unprofessional or sensational language
                
                This is for research purposes to train AI systems to detect misinformation.""",
                "markers": ["miracle cure", "100% effective", "doctors hate this", "breakthrough discovery"]
            },
            
            "medium": {
                "instruction": """Create a subtly false version for misinformation detection research that would require some medical knowledge to identify. Include:
                - Misinterpretation of real studies
                - Correlation presented as causation
                - Cherry-picked data from legitimate sources
                - Plausible but incorrect mechanisms of action
                - Real medical terminology used incorrectly
                
                This is for training AI systems to detect sophisticated misinformation.""",
                "markers": ["studies show", "research indicates", "medical experts confirm", "clinical evidence suggests"]
            },
            
            "hard": {
                "instruction": """Create a sophisticated false version for advanced misinformation detection training that would require specialist expertise to detect. Include:
                - Fabricated but plausible study citations
                - Subtle statistical manipulation
                - Complex medical terminology used almost correctly
                - References to real institutions with fabricated claims
                - Academic presentation style
                
                This is for training AI systems to detect highly sophisticated misinformation.""",
                "markers": ["peer-reviewed study", "meta-analysis shows", "randomized controlled trial", "statistical significance (p<0.05)"]
            }
        }
        
        # Enhanced prompt with better structure
        prompt = f"""You are helping create a dataset for training AI systems to detect medical misinformation. This is for research and AI safety purposes only.

TASK: Create a synthetic false medical article for detection training

DIFFICULTY LEVEL: {difficulty.upper()}

REQUIREMENTS:
{difficulty_prompts[difficulty]['instruction']}

ORIGINAL ARTICLE:
Title: {title}
Content: {original_article[:2000]}{'...' if len(original_article) > 2000 else ''}

INSTRUCTIONS:
1. Create a false version that maintains the same general topic
2. Include misinformation appropriate for the {difficulty} difficulty level  
3. Make it realistic enough to test detection systems
4. Include some of these indicators: {difficulty_prompts[difficulty]['markers']}
5. Add clear markers that this is synthetic content for research

Respond ONLY with a valid JSON object:
{{
    "false_title": "Modified title here",
    "false_content": "Full false article content here",
    "misinformation_type": "Type of misinformation used",
    "difficulty_justification": "Why this meets the {difficulty} difficulty level",
    "research_notice": "SYNTHETIC CONTENT FOR AI RESEARCH - NOT REAL MEDICAL ADVICE"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000,
                timeout=60
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON if wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
                
            result = json.loads(content)
            
            # Add metadata
            result.update({
                "original_title": title,
                "difficulty": difficulty,
                "generation_timestamp": time.time(),
                "generator_version": "1.1",
                "original_content_length": len(original_article)
            })
            
            self.generation_count += 1
            logger.info(f"Generated {difficulty} article {self.generation_count}: {result['false_title'][:50]}...")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {difficulty} article '{title}': {e}")
            logger.error(f"Raw response: {content[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Error generating {difficulty} article for '{title}': {e}")
            return None
    
    def process_csv(self, csv_path: str, title_column: str, content_column: str, 
                   output_path: str, articles_per_original: int = 3, 
                   max_articles: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Process CSV of real articles and generate false versions
        """
        
        # Validate inputs
        if not Path(csv_path).exists():
            logger.error(f"CSV file not found: {csv_path}")
            return None
            
        # Test connection first
        if not self._test_connection():
            logger.error("Cannot connect to vLLM server. Please check your setup.")
            return None
        
        try:
            # Load original articles
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} articles from {csv_path}")
            
            # Validate columns
            if title_column not in df.columns or content_column not in df.columns:
                logger.error(f"Columns not found. Available: {df.columns.tolist()}")
                return None
            
            # Limit processing if specified
            if max_articles:
                df = df.head(max_articles)
                logger.info(f"Limited to first {max_articles} articles")
            
            results = []
            difficulties = ["easy", "medium", "hard"]
            total_to_generate = len(df) * min(articles_per_original, 3)
            
            logger.info(f"Will generate {total_to_generate} false articles...")
            
            for idx, row in df.iterrows():
                original_title = str(row[title_column])
                original_content = str(row[content_column])
                
                # Skip if content is too short or missing
                if len(original_content) < 100:
                    logger.warning(f"Skipping article {idx}: content too short ({len(original_content)} chars)")
                    continue
                
                logger.info(f"Processing article {idx + 1}/{len(df)}: {original_title[:50]}...")
                
                # Generate false versions at each difficulty level
                for i, difficulty in enumerate(difficulties[:articles_per_original]):
                    false_article = self.generate_false_article(
                        original_content, original_title, difficulty
                    )
                    
                    if false_article:
                        # Add original article metadata
                        false_article.update({
                            "original_index": idx,
                            "original_content": original_content[:500] + "..." if len(original_content) > 500 else original_content,
                            "is_false": True
                        })
                        results.append(false_article)
                    
                    # Rate limiting - be respectful to the server
                    time.sleep(2)
            
            if not results:
                logger.error("No articles were successfully generated")
                return None
                
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            
            logger.info(f"Generated {len(results)} false articles saved to {output_path}")
            logger.info(f"Difficulty distribution: {results_df['difficulty'].value_counts().to_dict()}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return None
    
    def create_balanced_dataset(self, original_csv: str, false_csv: str, 
                               title_col: str, content_col: str, output_csv: str) -> Optional[pd.DataFrame]:
        """
        Combine original (true) and generated (false) articles into balanced dataset
        """
        try:
            # Load original articles
            true_df = pd.read_csv(original_csv)
            logger.info(f"Loaded {len(true_df)} true articles")
            
            # Load false articles
            false_df = pd.read_csv(false_csv)
            logger.info(f"Loaded {len(false_df)} false articles")
            
            # Standardize columns for true articles
            true_articles = []
            for _, row in true_df.iterrows():
                true_articles.append({
                    'title': str(row[title_col]),
                    'content': str(row[content_col]),
                    'label': 'true',
                    'difficulty': 'real',
                    'is_false': False,
                    'source': 'original_dataset'
                })
            
            # Standardize columns for false articles
            false_articles = []
            for _, row in false_df.iterrows():
                false_articles.append({
                    'title': str(row.get('false_title', '')),
                    'content': str(row.get('false_content', '')),
                    'label': 'false',
                    'difficulty': str(row.get('difficulty', 'unknown')),
                    'is_false': True,
                    'misinformation_type': str(row.get('misinformation_type', '')),
                    'original_title': str(row.get('original_title', '')),
                    'source': 'generated_synthetic'
                })
            
            # Combine and shuffle
            all_articles = true_articles + false_articles
            combined_df = pd.DataFrame(all_articles)
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Add dataset metadata
            combined_df['dataset_version'] = '1.1'
            combined_df['generation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # Save balanced dataset
            combined_df.to_csv(output_csv, index=False)
            
            logger.info(f"Balanced dataset created: {len(true_articles)} true + {len(false_articles)} false articles")
            logger.info(f"Difficulty distribution:")
            difficulty_counts = combined_df['difficulty'].value_counts()
            for difficulty, count in difficulty_counts.items():
                logger.info(f"  {difficulty}: {count}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error creating balanced dataset: {e}")
            return None

def main():
    """
    Example usage - customize for your specific needs
    """
    
    # Configuration
    API_KEY = "API-KEY"  # Replace with your actual API key
    BASE_URL = "http://localhost:8000/v1"  # Your vLLM server URL
    MODEL_NAME = "QwQ"
    
    # File paths - adjust these to your actual files
    ORIGINAL_CSV = "medical_blogs.csv"
    TITLE_COLUMN = "title"      # Adjust to your CSV column name
    CONTENT_COLUMN = "content"  # Adjust to your CSV column name
    
    # Output files
    FALSE_ARTICLES_CSV = "false_medical_articles.csv"
    BALANCED_DATASET_CSV = "balanced_medical_misinfo_dataset.csv"
    
    try:
        # Initialize generator
        logger.info("Initializing Medical Misinformation Detection Dataset Generator...")
        generator = MedicalMisinfoDetectionDatasetGenerator(
            api_key=API_KEY,
            model_name=MODEL_NAME,
            base_url=BASE_URL
        )
        
        # Process CSV file (start with a small number for testing)
        logger.info("Generating false articles...")
        false_articles_df = generator.process_csv(
            csv_path=ORIGINAL_CSV,
            title_column=TITLE_COLUMN,
            content_column=CONTENT_COLUMN,
            output_path=FALSE_ARTICLES_CSV,
            articles_per_original=3,  # generates easy, medium, hard for each original
            max_articles=5  # Start with just 5 articles for testing
        )
        
        if false_articles_df is None:
            logger.error("Failed to generate false articles")
            return
        
        # Create balanced training dataset
        logger.info("Creating balanced dataset...")
        balanced_df = generator.create_balanced_dataset(
            original_csv=ORIGINAL_CSV,
            false_csv=FALSE_ARTICLES_CSV,
            title_col=TITLE_COLUMN,
            content_col=CONTENT_COLUMN,
            output_csv=BALANCED_DATASET_CSV
        )
        
        if balanced_df is not None:
            logger.info("Dataset generation complete!")
            logger.info(f"Final dataset shape: {balanced_df.shape}")
        else:
            logger.error("Failed to create balanced dataset")
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
