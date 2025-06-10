import pandas as pd
import openai
from typing import List, Dict
import json
import time
import random

class MedicalMisinfoGenerator:
    def __init__(self, api_key: str, model_name: str = "QwQ"):
        """
        Initialize the generator with your vLLM-served QwQ model
        """
        self.client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=api_key
        )
        self.model_name = model_name
        
    def generate_false_article(self, original_article: str, title: str, difficulty: str) -> Dict:
        """
        Generate a false version of a medical article at specified difficulty level
        """
        
        # Define difficulty-specific prompts
        difficulty_prompts = {
            "easy": {
                "instruction": """Create a clearly false version of this medical article that contains obvious misinformation that would be easily spotted by any reader. Include:
                - Completely unsupported miracle cures
                - Obviously exaggerated statistics (like 100% cure rates)
                - Claims that contradict basic medical knowledge
                - Unprofessional language or sensational headlines""",
                "markers": ["miracle cure", "100% effective", "doctors hate this", "breakthrough"]
            },
            
            "medium": {
                "instruction": """Create a subtly false version that would require medical knowledge to identify as misinformation. Include:
                - Misinterpretation of real studies
                - Correlation presented as causation
                - Cherry-picked data from legitimate sources
                - Plausible but incorrect mechanisms of action
                - Real medical terminology used incorrectly""",
                "markers": ["studies show", "research indicates", "medical experts", "clinical trial"]
            },
            
            "hard": {
                "instruction": """Create a sophisticated false version that would require specialist expertise to detect. Include:
                - Fabricated but plausible study citations
                - Subtle statistical manipulation
                - Complex medical terminology used almost correctly
                - References to real institutions with fake claims
                - Sophisticated presentation mimicking real medical literature""",
                "markers": ["peer-reviewed", "meta-analysis", "randomized controlled trial", "statistical significance"]
            }
        }
        
        prompt = f"""
        You are tasked with creating synthetic false medical news for misinformation detection research.
        
        DIFFICULTY LEVEL: {difficulty.upper()}
        
        {difficulty_prompts[difficulty]['instruction']}
        
        ORIGINAL ARTICLE:
        Title: {title}
        Content: {original_article}
        
        Create a false version that:
        1. Maintains the same general topic and structure
        2. Contains medical misinformation at the {difficulty} level
        3. Feels realistic enough to potentially fool readers at this difficulty level
        4. Includes some of these markers: {difficulty_prompts[difficulty]['markers']}
        
        Respond with a JSON object containing:
        {{
            "false_title": "...",
            "false_content": "...",
            "misinformation_type": "...",
            "difficulty_justification": "..."
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            result["original_title"] = title
            result["difficulty"] = difficulty
            result["generation_timestamp"] = time.time()
            
            return result
            
        except Exception as e:
            print(f"Error generating {difficulty} article for '{title}': {e}")
            return None
    
    def process_csv(self, csv_path: str, title_column: str, content_column: str, 
                   output_path: str, articles_per_original: int = 3):
        """
        Process CSV of real articles and generate false versions
        """
        # Load original articles
        df = pd.read_csv(csv_path)
        
        results = []
        difficulties = ["easy", "medium", "hard"]
        
        for idx, row in df.iterrows():
            original_title = row[title_column]
            original_content = row[content_column]
            
            print(f"Processing article {idx + 1}/{len(df)}: {original_title[:50]}...")
            
            # Generate false versions at each difficulty level
            for difficulty in difficulties[:articles_per_original]:
                false_article = self.generate_false_article(
                    original_content, original_title, difficulty
                )
                
                if false_article:
                    # Add original article metadata
                    false_article.update({
                        "original_index": idx,
                        "original_content": original_content,
                        "is_false": True
                    })
                    results.append(false_article)
                
                # Rate limiting
                time.sleep(1)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        print(f"Generated {len(results)} false articles saved to {output_path}")
        return results_df
    
    def create_balanced_dataset(self, original_csv: str, false_csv: str, 
                               title_col: str, content_col: str, output_csv: str):
        """
        Combine original (true) and generated (false) articles into balanced dataset
        """
        # Load original articles
        true_df = pd.read_csv(original_csv)
        true_df['is_false'] = False
        true_df['difficulty'] = 'real'
        true_df['label'] = 'true'
        
        # Load false articles
        false_df = pd.read_csv(false_csv)
        false_df['label'] = 'false'
        
        # Standardize columns
        true_articles = []
        for _, row in true_df.iterrows():
            true_articles.append({
                'title': row[title_col],
                'content': row[content_col],
                'label': 'true',
                'difficulty': 'real',
                'is_false': False
            })
        
        false_articles = []
        for _, row in false_df.iterrows():
            false_articles.append({
                'title': row['false_title'],
                'content': row['false_content'],
                'label': 'false',
                'difficulty': row['difficulty'],
                'is_false': True,
                'misinformation_type': row.get('misinformation_type', ''),
                'original_title': row.get('original_title', '')
            })
        
        # Combine and shuffle
        all_articles = true_articles + false_articles
        combined_df = pd.DataFrame(all_articles)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        # Save balanced dataset
        combined_df.to_csv(output_csv, index=False)
        
        print(f"Balanced dataset created with {len(true_articles)} true and {len(false_articles)} false articles")
        print(f"Difficulty distribution:")
        print(combined_df['difficulty'].value_counts())
        
        return combined_df

# Usage example
def main():
    # Initialize generator
    generator = MedicalMisinfoGenerator(api_key="your-api-key")
    
    # Process your CSV file
    false_articles_df = generator.process_csv(
        csv_path="medical_blogs.csv",
        title_column="title",  # adjust to your CSV column names
        content_column="content",  # adjust to your CSV column names
        output_path="false_medical_articles.csv",
        articles_per_original=3  # generates easy, medium, hard for each original
    )
    
    # Create balanced training dataset
    balanced_df = generator.create_balanced_dataset(
        original_csv="medical_blogs.csv",
        false_csv="false_medical_articles.csv",
        title_col="title",
        content_col="content",
        output_csv="balanced_medical_misinfo_dataset.csv"
    )
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()