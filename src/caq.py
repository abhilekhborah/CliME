import torch
import spacy
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

class ClimateAlignmentQuotient:
    def __init__(self, use_gpu=True, debug=False):
        """
        Initialize the Climate Actionability Quotient calculator.

        Args:
            use_gpu: Whether to use GPU for model inference
            debug: Enable debug output
        """
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # Load NLP models
        print("Loading models...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("Successfully loaded SpaCy model")
        except Exception as e:
            print(f"Error loading SpaCy model: {e}")
            self.nlp = None

        # Load sentence transformer model for semantic coherence
        try:
            self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
            self.sentence_model.to(self.device)
            print("Successfully loaded Sentence Transformer model")
        except Exception as e:
            print(f"Error loading Sentence Transformer model: {e}")
            self.sentence_model = None

        # Define models configuration
        self.models_config = {
            "detector": {
                "model": "climatebert/distilroberta-base-climate-detector",
                "tokenizer": "climatebert/distilroberta-base-climate-detector"
            },
            "transition_physical": {
                "model": "climatebert/transition-physical",
                "tokenizer": "climatebert/distilroberta-base-climate-detector"
            },
            "evidence": {
                "model": "climate-nlp/longformer-large-4096-1-detect-evidence",
                "tokenizer": "climate-nlp/longformer-large-4096-1-detect-evidence"
            },
            "specificity": {
                "model": "climatebert/distilroberta-base-climate-specificity",
                "tokenizer": "climatebert/distilroberta-base-climate-specificity"
            }
        }

        # Initialize containers for models and tokenizers
        self.models = {}
        self.tokenizers = {}

        # Load ClimateBERT models
        for name, config in self.models_config.items():
            model_path = config["model"]
            tokenizer_path = config["tokenizer"]

            print(f"Loading {name} model...")

            try:
                self.tokenizers[name] = AutoTokenizer.from_pretrained(tokenizer_path)
                self.models[name] = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.models[name].to(self.device)
                self.models[name].eval()
                print(f"Successfully loaded {name} model")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
                self.models[name] = None
                self.tokenizers[name] = None

        # Component weights for the final CAQ calculation
        self.weights = {
            "detector": 0.25,      # ClimateBERT detection score
            "as": 0.3,            # Action Articulation Score
            "transition_physical": 0.15,  # Transition physical score
            "evidence": 0.2,     # Sentiment score
            "specificity": 0.1   # Specificity score
        }

    def _preprocess_text(self, text):
        """
        Basic preprocessing for text analysis.

        Args:
            text: Input text

        Returns:
            Processed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if self.debug:
            print(f"Preprocessed text: {text[:100]}...")

        return text

    def get_bert_score(self, text, model_name):
        """
        Get prediction score from a ClimateBERT model.

        Args:
            text: Text to evaluate
            model_name: Name of the model to use

        Returns:
            Prediction score
        """
        if not self.models.get(model_name) or not self.tokenizers.get(model_name):
            print(f"Warning: {model_name} model not available, returning default score")
            return 0

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

            # Most models use index 1 for positive class, but we check dimensions
            if probabilities.shape[1] == 2:
                # Binary classification: use positive class (index 1)
                score = probabilities[0, 1].item()
            else:
                # Multi-class: use average of all positive signals
                # Exclude first class (often neutral or negative)
                score = torch.mean(probabilities[0, 1:]).item()

        # For the sentiment model, map the raw probability to a static score
        if model_name == "evidence":
            score = self.map_sentiment_probability(score)

        if self.debug:
            print(f"{model_name} score: {score:.4f}")

        return score

    def map_sentiment_probability(self, prob):
        """
        Map a sentiment probability to a static score.
        
        Rules:
          - If prob < 0.5: return 0.1.
          - If prob == 0.5: return 0.5.
          - If prob > 0.95: return 1.0.
          - If 0.5 < prob <= 0.95: divide the interval into four equal parts and return:
                (0.5, 0.6125]   -> 0.6
                (0.6125, 0.725] -> 0.7
                (0.725, 0.8375] -> 0.8
                (0.8375, 0.95]  -> 0.9
        """
        return prob

    def calculate_articulation_score(self, text):
        """
        Calculate an improved Articulation Score - a comprehensive metric evaluating how effectively
        content is articulated across multiple dimensions of communication quality.

        Evaluates clarity, coherence, completeness, readability, and engagement factors.

        Args:
            text: Text to evaluate

        Returns:
            Improved articulation score (0-1)
        """
        if not self.nlp:
            print("Warning: SpaCy not available, returning default articulation score")
            return 0

        # Parse the text
        doc = self.nlp(text)
        sentences = list(doc.sents)
        total_sentences = len(sentences)

        if total_sentences == 0:
            return 0  # Default for empty text

        # 1. COHERENCE (Combined Syntactic and Semantic)
        # 1a. Syntactic coherence through discourse markers
        discourse_markers = 0
        for token in doc:
            if token.pos_ in ["CCONJ", "SCONJ"] or token.dep_ == "mark":
                discourse_markers += 1

        syntactic_coherence = min(1.0, discourse_markers / (total_sentences * 0.5))

        # 1b. Semantic coherence through sentence similarity
        semantic_coherence = 0.5  # Default value
        if self.sentence_model and total_sentences > 1:
            try:
                # Get embeddings for each sentence
                sentence_texts = [sent.text for sent in sentences]
                embeddings = self.sentence_model.encode(sentence_texts)

                # Calculate similarity between adjacent sentences
                similarities = []
                for i in range(len(embeddings) - 1):
                    similarity = self.cosine_similarity(embeddings[i], embeddings[i+1])
                    similarities.append(similarity)

                semantic_coherence = sum(similarities) / len(similarities)
            except Exception as e:
                if self.debug:
                    print(f"Error calculating semantic coherence: {e}")

        # Combined coherence score
        coherence_score = 0.6 * syntactic_coherence + 0.4 * semantic_coherence

        # 2. COMPLETENESS (Adjusted for imperative sentences)
        complete_statements = 0
        imperative_statements = 0

        for sent in sentences:
            # Check for complete subject-predicate structure
            has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] for token in sent)
            has_predicate = any(token.pos_ == "VERB" and token.dep_ == "ROOT" for token in sent)

            if has_subject and has_predicate:
                complete_statements += 1
            # Check for imperative sentences (verb at start without subject)
            elif len(sent) > 0 and sent[0].pos_ == "VERB":
                imperative_statements += 1

        # Consider both regular complete and imperative statements
        completeness_score = (complete_statements + imperative_statements) / total_sentences

        # Calculate weighted final score (redistributed weights after removing clarity and readability)
        articulation_score = (
            0.50 * coherence_score +
            0.50 * completeness_score
        )

        if self.debug:
            print(f"Articulation Score: {articulation_score:.4f}")
            print(f"  - Coherence: {coherence_score:.4f} (Syntactic: {syntactic_coherence:.2f}, Semantic: {semantic_coherence:.2f})")
            print(f"  - Completeness: {completeness_score:.4f} (Complete: {complete_statements}, Imperative: {imperative_statements})")

        return articulation_score

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

    def calculate_caq(self, text):
        """
        Calculate the Climate Actionability Quotient (CAQ) using the five core metrics.

        Args:
            text: Climate-related text to evaluate

        Returns:
            Dictionary containing individual component scores and the composite CAQ score
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Calculate individual component scores
        component_scores = {
            # ClimateBERT model scores
            "detector": self.get_bert_score(processed_text, "detector"),
            "transition_physical": self.get_bert_score(processed_text, "transition_physical"),
            "evidence": self.get_bert_score(processed_text, "evidence"),
            "specificity": self.get_bert_score(processed_text, "specificity"),

            # Articulation Score
            "as": self.calculate_articulation_score(processed_text)
        }

        # Calculate weighted CAQ score
        caq_score = sum(component_scores[metric] * self.weights[metric]
                        for metric in component_scores)

        # Create human-readable component names for output
        readable_components = {
            "detector": "Resonance",
            "as": "Articulation",
            "transition_physical": "Transition",
            "evidence": "Evidence",
            "specificity": "Specificity",
            "caq": "Alignment"
        }

        # Add CAQ score to component scores
        component_scores["caq"] = caq_score

        # Return results
        results = {
            "component_scores": component_scores,
            "readable_components": readable_components,
            "weights": self.weights
        }

        if self.debug:
            print(f"CAQ Score: {caq_score:.4f}")

        return results


