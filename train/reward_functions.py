import logging
import os
from datetime import datetime
import re
import random
import torch
import numpy as np
from typing import List, Optional, Tuple, Any


def _fallback_language_match(text, query):
    """
    Fallback language matching method using basic character-set heuristics.
    Used when langid is not available.
    """
    # Character sets that are distinctive for different scripts
    script_markers = {
        'devanagari': [chr(c) for c in range(0x0900, 0x097F)],  # Hindi, Sanskrit, etc.
        'arabic': [chr(c) for c in range(0x0600, 0x06FF)],      # Arabic, Persian, Urdu, etc.
        'chinese': [chr(c) for c in range(0x4E00, 0x9FFF)],     # Chinese
        'cyrillic': [chr(c) for c in range(0x0400, 0x04FF)],    # Russian, etc.
        'thai': [chr(c) for c in range(0x0E00, 0x0E7F)],        # Thai
        'hangul': [chr(c) for c in range(0xAC00, 0xD7AF)],      # Korean
        'latin': [chr(c) for c in range(0x0041, 0x007A)]        # Latin/Roman (English, Spanish, etc.)
    }
    
    # Count script markers in query and text
    query_scripts = {}
    text_scripts = {}
    
    for script, markers in script_markers.items():
        # Check which script markers appear in query and text
        query_count = sum(1 for c in query if c in markers)
        text_count = sum(1 for c in text if c in markers)
        
        query_scripts[script] = query_count
        text_scripts[script] = text_count
    
    # Determine dominant script in query
    dominant_query_script = max(query_scripts.items(), key=lambda x: x[1])[0]
    if query_scripts[dominant_query_script] < 3:
        # If minimal script markers, default to 'latin' (most common)
        dominant_query_script = 'latin'
    
    # Check if the same script is dominant in the text
    if dominant_query_script == max(text_scripts.items(), key=lambda x: x[1])[0]:
        # Scripts match, but how strongly?
        query_script_ratio = query_scripts[dominant_query_script] / max(1, len(query))
        text_script_ratio = text_scripts[dominant_query_script] / max(1, len(text))
        
        # The reward is the ratio of matching script markers in the text
        # scaled by how confident we are about the query's script
        reward = text_script_ratio * min(1.0, query_script_ratio * 2)
        return max(0.3, reward)  # At least 0.3 for matching script
    else:
        # Scripts don't match
        return 0.1  # Small non-zero reward


def _check_script_match(text1, text2):
    """
    Check if two texts use the same script/character set.
    Returns a similarity score between 0 and 1.
    """
    # Character sets that are distinctive for different scripts
    script_markers = {
        'devanagari': [chr(c) for c in range(0x0900, 0x097F)],  # Hindi, Sanskrit, etc.
        'arabic': [chr(c) for c in range(0x0600, 0x06FF)],      # Arabic, Persian, Urdu, etc.
        'chinese': [chr(c) for c in range(0x4E00, 0x9FFF)],     # Chinese
        'cyrillic': [chr(c) for c in range(0x0400, 0x04FF)],    # Russian, etc.
        'thai': [chr(c) for c in range(0x0E00, 0x0E7F)],        # Thai
        'hangul': [chr(c) for c in range(0xAC00, 0xD7AF)],      # Korean
        'latin': [chr(c) for c in range(0x0041, 0x007A)]        # Latin/Roman (English, Spanish, etc.)
    }
    
    # Count script markers in each text
    text1_scripts = {}
    text2_scripts = {}
    
    for script, markers in script_markers.items():
        # Count markers in each text
        text1_count = sum(1 for c in text1 if c in markers)
        text2_count = sum(1 for c in text2 if c in markers)
        
        text1_scripts[script] = text1_count
        text2_scripts[script] = text2_count
    
    # Find dominant script in each text
    dominant_script1 = max(text1_scripts.items(), key=lambda x: x[1])[0]
    dominant_script2 = max(text2_scripts.items(), key=lambda x: x[1])[0]
    
    # If minimal script markers, default to 'latin'
    if text1_scripts[dominant_script1] < 3:
        dominant_script1 = 'latin'
    if text2_scripts[dominant_script2] < 3:
        dominant_script2 = 'latin'
    
    # Calculate match score
    if dominant_script1 == dominant_script2:
        return 1.0  # Perfect script match
    else:
        # Check for partial match (some characters in the same script)
        # This handles code-switching/mixed language content
        script1_in_text2 = text2_scripts.get(dominant_script1, 0) / max(1, len(text2))
        script2_in_text1 = text1_scripts.get(dominant_script2, 0) / max(1, len(text1))
        
        return max(script1_in_text2, script2_in_text1)


########################
# Reward functions
########################

def format_reward_func_factual(completions, target=None, prompts=None, **kwargs):
    """
    Checks whether completion follows the format: <think>...</think>\n<answer>...</answer>
    
    Args:
        completions (list[str])
        target (list[str]) - unused but kept for API compatibility

    Returns:
        list[float]
    """
    rewards = []
    for completion in completions:
        try:
            completion = "<think>" + completion  # pre-add <think> if you assume assistant always starts inside thinking
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
                
            # Log samples occasionally
            if random.random() < 0.05:  # 5% chance to log samples
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "factual_samples_v2.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        except Exception:
            rewards.append(0.0)
    return rewards



def correctness_reward_func_factual(completions, targets=None, prompts=None, **kwargs):
    """
    Checks whether extracted <answer> matches the normalized ground truth target.
    Normalizes both prediction and target using version_dict.

    Args:
        completions (list[str])
        targets (list[str])

    Returns:
        list[float]: Reward scores
    """
    rewards = []

    
    targets = kwargs['target']
        
    version_dict = {
        # English mappings
        "usa": "united states",
        "u.s.a.": "united states",
        "united states of america": "united states",
        "america": "united states",
        "bharat": "india",
        "hindustan": "india",
        "prc": "china",
        "people's republic of china": "china",
        "russian federation": "russia",
        "ksa": "saudi arabia",
        "kingdom of saudi arabia": "saudi arabia",
        "french republic": "france",
        "federal democratic republic of nepal": "nepal",
        "hellenic republic": "greece",
        "turkiye cumhuriyeti": "turkey",
        "republic of kenya": "kenya",
        "kingdom of thailand": "thailand",
        # Hindi mappings
        "अमेरिका": "संयुक्त राज्य अमेरिका",
        # you can expand more mappings
    }

    for completion, target in zip(completions, targets):
        try:
            completion = "<think>" + completion

            # Extract <answer>...</answer>
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue

            prediction = match.group(1).strip().lower()
            target_normalized = target.strip().lower()

            # Normalize both prediction and target
            prediction_normalized = version_dict.get(prediction, prediction)
            target_normalized = version_dict.get(target_normalized, target_normalized)

            if random.random() < 0.01:  # 1% chance to log predictions
                print(f"Prediction: {prediction_normalized}, Target: {target_normalized}")

            if prediction_normalized == target_normalized:
                rewards.append(1.0)
                
                # Log successful samples occasionally
                if random.random() < 0.10:  # 10% chance to log successful samples
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "successful_factual_samples_v2.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"Completion: {completion}\n")
                        f.write(f"Prediction: {prediction_normalized}, Target: {target_normalized}\n")
            else:
                rewards.append(0.0)

        except Exception as e:
            print(f"Error in reward function: {e}")
            rewards.append(0.0)

    return rewards


from difflib import SequenceMatcher

def correctness_reward_func_factual_fuzzy(completions, targets=None, prmopts=None, **kwargs):
    """
    Checks whether extracted <answer> matches any answer in the answer_list,
    with flexible parsing and comparison of answers.
    
    Args:
        completions (list[str]): Model completions
        targets (list[str]): Not used, targets come from kwargs['target']
        **kwargs: Contains 'target' (primary answers) and 'answer_list' (all acceptable answers)
        
    Returns:
        list[float]: Reward scores between 0.0 and 1.0, with partial credit for close matches
    """
    rewards = []
    
    # Get targets and answer_lists from kwargs
    targets = kwargs.get('target', [])
    answer_lists = kwargs.get('answer_list', [])
    
    # Define thresholds for fuzzy matching
    exact_match_threshold = 1.0
    high_similarity_threshold = 0.9
    medium_similarity_threshold = 0.75
    low_similarity_threshold = 0.6
    
    def normalize_text(text):
        """Normalize text by removing extra spaces, lowercasing, etc."""
        if not isinstance(text, str):
            text = str(text)
        # Remove extra whitespace and lowercase
        return re.sub(r'\s+', ' ', text).strip().lower()
    
    def similarity_ratio(text1, text2):
        """Calculate similarity ratio between two strings using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def contains_key_elements(prediction, acceptable):
        """Check if prediction contains all important words from acceptable answer."""
        # Tokenize and filter for meaningful words (longer than 3 chars)
        acceptable_words = [w for w in acceptable.split() if len(w) > 3]
        if not acceptable_words:  # If no meaningful words, fall back to all words
            acceptable_words = acceptable.split()
        
        # Count how many key words are found in the prediction
        found_words = sum(1 for word in acceptable_words if word in prediction)
        
        # Calculate coverage ratio
        if len(acceptable_words) == 0:
            return 0.0
        return found_words / len(acceptable_words)
    
    def is_numeric_match(pred, target, tolerance=0.05):
        """Check if numerical values match within tolerance."""
        try:
            # Extract numbers from strings
            pred_nums = [float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+', pred)]
            target_nums = [float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+', target)]
            
            # If both have exactly one number, compare them
            if len(pred_nums) == 1 and len(target_nums) == 1:
                return abs(pred_nums[0] - target_nums[0]) <= tolerance * max(1, abs(target_nums[0]))
            
            # If they have the same number of numerical values, compare each pair
            if len(pred_nums) == len(target_nums) and len(pred_nums) > 0:
                matches = 0
                for p, t in zip(sorted(pred_nums), sorted(target_nums)):
                    if abs(p - t) <= tolerance * max(1, abs(t)):
                        matches += 1
                return matches / len(target_nums)
        except:
            pass
        return 0.0
    
    for i, (completion, target) in enumerate(zip(completions, targets)):
        try:
            # Ensure the completion has the <think> tag for proper pattern matching
            if "<think>" not in completion:
                completion = "<think>" + completion
            
            # Extract the answer text between <answer> tags
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue
            
            # Get the predicted answer and normalize it
            prediction = normalize_text(match.group(1))
            
            # Get all possible acceptable answers
            all_acceptable_answers = []
            
            # Always include the primary target
            if target:
                all_acceptable_answers.append(normalize_text(target))
            
            # Process the answer_list
            if i < len(answer_lists):
                current_answer_list = answer_lists[i]
                
                # Handle different formats of answer_list
                if isinstance(current_answer_list, str):
                    # Try parsing as a comma-separated string first
                    if "," in current_answer_list:
                        parts = [normalize_text(p) for p in current_answer_list.split(",")]
                        all_acceptable_answers.extend(parts)
                    else:
                        all_acceptable_answers.append(normalize_text(current_answer_list))
                
                elif isinstance(current_answer_list, list):
                    # Process each answer in the list
                    for ans in current_answer_list:
                        if isinstance(ans, str):
                            # If it contains commas, split it
                            if "," in ans:
                                parts = [normalize_text(p) for p in ans.split(",")]
                                all_acceptable_answers.extend(parts)
                            else:
                                all_acceptable_answers.append(normalize_text(ans))
                        else:
                            # For non-string answers (e.g., numbers)
                            all_acceptable_answers.append(normalize_text(str(ans)))
            
            # Remove duplicates from acceptable answers
            all_acceptable_answers = list(set(all_acceptable_answers))
            
            # Debug logging for a small percentage of examples
            if random.random() < 0.01:
                print(f"\nPrediction: '{prediction}'")
                print(f"All acceptable answers: {all_acceptable_answers}")
            
            # Check for matches with various degrees of flexibility
            best_match_score = 0.0
            best_match_answer = ""
            match_method = ""
            
            for acceptable in all_acceptable_answers:
                # Try exact match first (original behavior)
                if prediction == acceptable:
                    best_match_score = 1.0
                    best_match_answer = acceptable
                    match_method = "exact"
                    break
                
                # Calculate similarity score
                sim_score = similarity_ratio(prediction, acceptable)
                
                # Check for numeric match if we have numbers
                num_match_score = is_numeric_match(prediction, acceptable)
                
                # Check for key elements containment
                containment_score = contains_key_elements(prediction, acceptable)
                
                # Take the best score from the different methods
                method_scores = [
                    (sim_score, "similarity"),
                    (num_match_score, "numeric"),
                    (containment_score, "key_elements")
                ]
                
                # Get the highest score and its method
                current_score, current_method = max(method_scores, key=lambda x: x[0])
                
                # Update best match if this is better
                if current_score > best_match_score:
                    best_match_score = current_score
                    best_match_answer = acceptable
                    match_method = current_method
            
            # Assign reward based on match quality
            if best_match_score >= exact_match_threshold:
                reward = 1.0
            elif best_match_score >= high_similarity_threshold:
                reward = 0.9
            elif best_match_score >= medium_similarity_threshold:
                reward = 0.7
            elif best_match_score >= low_similarity_threshold:
                reward = 0.5
            else:
                reward = 0.0
            
            rewards.append(reward)
            
            # Log samples for analysis
            if random.random() < 0.05:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = "successful_factual_samples_v2.txt" if reward >= 0.9 else "partial_match_samples_v2.txt"
                
                with open(f"completion_samples/{log_file}", "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(f"Completion: {completion}\n")
                    f.write(f"Prediction: '{prediction}'\n")
                    f.write(f"Best Match: '{best_match_answer}' (Score: {best_match_score:.2f}, Method: {match_method})\n")
                    f.write(f"Reward: {reward}\n")
                    f.write(f"All acceptable answers: {all_acceptable_answers}\n")
            
            # Log failures occasionally for debugging
            if reward == 0.0 and random.random() < 0.05:
                os.makedirs("completion_samples", exist_ok=True)
                with open("completion_samples/failed_matches_v2.txt", "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(f"Completion: {completion}\n")
                    f.write(f"Prediction: '{prediction}'\n")
                    f.write(f"Failed to match. Best match: '{best_match_answer}' (Score: {best_match_score:.2f})\n")
                    f.write(f"Acceptable answers were: {all_acceptable_answers}\n")
                    
                    # Log individual similarity scores for analysis
                    f.write("Detailed match scores:\n")
                    for acceptable in all_acceptable_answers:
                        sim = similarity_ratio(prediction, acceptable)
                        num = is_numeric_match(prediction, acceptable)
                        key = contains_key_elements(prediction, acceptable)
                        f.write(f"  - '{acceptable}': Similarity={sim:.2f}, Numeric={num:.2f}, KeyElements={key:.2f}\n")
            
        except Exception as e:
            print(f"Error in reward function: {e}")
            rewards.append(0.0)
    
    return rewards


############

def language_matching_reward_func(completions, queries, prompts=None, **kwargs):
    """
    Checks whether tokens inside <think></think> are in the same language as the query.
    Awards higher rewards for higher percentage of matching language tokens.
    
    Args:
        completions (list[str]): Model completions containing <think></think> sections
        queries (list[str]): The original queries, used to detect language
        
    Returns:
        list[float]: Reward scores between 0.0 and 1.0
    """
    import re
    import os
    import random
    
    # Import langid with error handling
    try:
        import langid
        # Initialize langid for language detection with common languages
        langid.set_languages(['en', 'hi', 'sw', 'zh', 'es', 'fr', 'ar', 'ru', 'ja', 'de', 'bn', 'pt'])
        langid_available = True
    except ImportError:
        print("Warning: langid not available. Install with 'pip install langid'")
        langid_available = False
    
    rewards = []
    
    for completion, query in zip(completions, queries):
        try:
            # Add <think> prefix if not present (as in your original function)
            if not completion.startswith("<think>"):
                completion = "<think>" + completion
            
            # Extract text between <think> and </think> tags
            think_match = re.search(r"<think>([\s\S]*?)<\/think>", completion, re.DOTALL)
            
            if think_match is None:
                rewards.append(0.0)
                continue
                
            think_content = think_match.group(1).strip()
            
            # If thinking content is too short, can't properly evaluate
            if len(think_content) < 10:
                rewards.append(0.2)  # Small non-zero reward for valid format at least
                continue
            
            # If langid is not available, fall back to simple heuristics
            if not langid_available:
                reward = _fallback_language_match(think_content, query)
                rewards.append(reward)
                continue
            
            # Detect language of query
            query_lang, query_conf = langid.classify(query)
            
            # For longer thinking content, break into chunks for better language detection
            chunk_size = 100  # Characters per chunk
            chunks = [think_content[i:i+chunk_size] for i in range(0, len(think_content), chunk_size)]
            
            # Detect language for each chunk
            matching_chunks = 0
            total_chunks = len(chunks)
            chunk_langs = []
            
            for chunk in chunks:
                if len(chunk.strip()) < 10:  # Skip very small chunks
                    total_chunks -= 1
                    continue
                    
                chunk_lang, confidence = langid.classify(chunk)
                chunk_langs.append(chunk_lang)
                
                if chunk_lang == query_lang:
                    matching_chunks += 1
            
            # Calculate percentage of matching language chunks
            if total_chunks == 0:
                # If no valid chunks, give neutral score
                reward = 0.5
            else:
                # The reward is the percentage of chunks in the query language
                reward = matching_chunks / total_chunks
            
            # Add a small bonus for any matching at all (to avoid zero rewards for partial matches)
            if matching_chunks > 0 and reward < 0.1:
                reward = 0.1
                
            # Log samples occasionally for debugging
            if random.random() < 0.01:  # 1% chance to log
                try:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "language_match_samples_v2.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"Query ({query_lang}): {query}\n")
                        f.write(f"Think content (sample langs: {chunk_langs[:3]}...): {think_content[:100]}...\n")
                        f.write(f"Matching chunks: {matching_chunks}/{total_chunks}\n")
                        f.write(f"Reward: {reward}\n")
                except Exception as e:
                    print(f"Warning: Could not write log: {e}")
            
            rewards.append(reward)
            
        except Exception as e:
            print(f"Error in language matching reward function: {e}")
            # Fallback to basic heuristics when errors occur
            try:
                fallback_reward = _fallback_language_match(think_content, query)
                rewards.append(fallback_reward)
            except:
                rewards.append(0.5)  # Neutral reward on complete failure
            
    return rewards


# More resilient advanced version that handles code-switching and language mixtures
def language_matching_reward_func_advanced(completions, targets=None, prompts=None, **kwargs):
    """
    Advanced version that handles code-switching and mixture of languages.
    Uses simpler sentence splitting and multiple language detection methods.
    
    Args:
        completions (list[str]): Model completions containing <think></think> sections
        queries (list[str]): The original queries, used to detect language
        
    Returns:
        list[float]: Reward scores between 0.0 and 1.0
    """
    
    if 'translated_question' in kwargs:
        queries = kwargs['translated_question']
    else:
        queries = kwargs['question']
    # Try multiple language detection libraries with fallbacks
    langdetect_available = False
    langid_available = False
    
    try:
        import langdetect
        from langdetect import detect_langs
        langdetect_available = True
    except ImportError:
        print("Warning: langdetect not available. Install with 'pip install langdetect'")
    
    langdetect_available = False # not use langdetect for consistency with Mathematical Reasoning

    if not langdetect_available:
        try:
            import langid
            langid.set_languages(['en', 'hi', 'sw', 'zh', 'el', 'fr', 'ar', 'ru', 'ja', 'ne', 'tr', 'th'])
            langid_available = True
        except ImportError:
            print("Warning: Neither langdetect nor langid available. Install with 'pip install langdetect langid'")
    
    # Simple sentence splitting function that doesn't require NLTK
    def simple_sentence_split(text):
        # Split by common sentence terminators
        sentences = []
        for rough_sentence in re.split(r'(?<=[.!?])\s+', text):
            # Further split by newlines, which often indicate sentence breaks in many formats
            for s in rough_sentence.split('\n'):
                s = s.strip()
                if s:  # Only add non-empty sentences
                    sentences.append(s)
        return sentences
    
    rewards = []
    for completion, query in zip(completions, queries):
        try:
            # Add <think> prefix if not present
            if not completion.startswith("<think>"):
                completion = "<think>" + completion
            
            # Extract text between <think> and </think> tags
            think_match = re.search(r"<think>([\s\S]*?)<\/think>", completion, re.DOTALL)
            
            if think_match is None:
                rewards.append(0.0)
                continue
                
            think_content = think_match.group(1).strip()
            
            # If thinking content is too short, can't properly evaluate
            if len(think_content) < 10:
                rewards.append(0.2)  # Small non-zero reward for valid format
                continue
            
            # Detect language of query
            primary_query_lang = None
            if langdetect_available:
                try:
                    query_langs = langdetect.detect_langs(query)
                    primary_query_lang = query_langs[0].lang
                    primary_query_prob = query_langs[0].prob
                except:
                    primary_query_lang = None
            
            if primary_query_lang is None and langid_available:
                try:
                    primary_query_lang, _ = langid.classify(query)
                except:
                    primary_query_lang = None
            
            # If we still can't detect language, use fallback method
            if primary_query_lang is None:
                # Use the fallback method from the basic function
                reward = _fallback_language_match(think_content, query)
                rewards.append(reward)
                continue
            
            # Break thinking content into sentences
            sentences = simple_sentence_split(think_content)
            
            # Skip if no valid sentences
            if not sentences:
                rewards.append(0.3)  # Some reward for valid format
                continue
                
            matching_sentences = 0
            total_sentences = len(sentences)
            
            # For each sentence, check language alignment with query
            for sentence in sentences:
                if len(sentence.strip()) < 5:  # Skip very short sentences
                    total_sentences -= 1
                    continue
                
                sentence_lang = None
                
                # Try langdetect first if available
                if langdetect_available:
                    try:
                        sent_langs = langdetect.detect_langs(sentence)
                        for lang_prob in sent_langs:
                            if lang_prob.lang == primary_query_lang and lang_prob.prob > 0.3:
                                matching_sentences += 1
                                break
                        continue  # If we got here, langdetect worked, so continue to next sentence
                    except:
                        pass  # Fall through to next method if langdetect fails
                
                # Try langid if langdetect not available or failed
                if langid_available:
                    try:
                        sent_lang, _ = langid.classify(sentence)
                        if sent_lang == primary_query_lang:
                            matching_sentences += 1
                        continue  # If we got here, langid worked, so continue to next sentence
                    except:
                        pass  # Fall through to fallback if both methods fail
                
                # If both methods failed, use character-based heuristic
                script_match = _check_script_match(sentence, query)
                if script_match > 0.5:  # Significant script match
                    matching_sentences += 1
            
            # Calculate final reward
            if total_sentences == 0:
                reward = 0.4  # Neutral-positive score if no valid sentences
            else:
                # Base reward on percentage of matching sentences
                reward = matching_sentences / total_sentences
            
            # Log samples occasionally
            if random.random() < 0.01:  # 1% chance to log
                try:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "language_match_advanced_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"Query lang: {primary_query_lang}\n")
                        f.write(f"Query: {query}\n")
                        f.write(f"Think content (sample): {think_content[:150]}...\n")
                        f.write(f"Matching sentences: {matching_sentences}/{total_sentences}\n")
                        f.write(f"Reward: {reward:.7f}\n")  # Updated to 7 decimal places
                except Exception as e:
                    print(f"Warning: Could not write log: {e}")
            
            # Log detailed reward information occasionally
            if random.random() < 0.01:  # 1% of batches
                try:
                    os.makedirs("logs/reward_details", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    log_file = os.path.join("logs/reward_details", f"language_reward_{timestamp}_v2.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n==============\n")
                        f.write(f"Query lang: {primary_query_lang}\n")
                        f.write(f"Reward: {reward:.7f}\n")  # Using 7 decimal places for higher precision
                        f.write(f"Matching sentences: {matching_sentences}/{total_sentences}\n")
                except Exception as e:
                    print(f"Warning: Could not write detailed reward log: {e}")
            
            rewards.append(reward)
            
        except Exception as e:
            print(f"Error in advanced language matching reward function: {e}")
            # Use fallback method on error
            try:
                fallback_reward = _fallback_language_match(think_content, query)
                rewards.append(fallback_reward)
            except:
                rewards.append(0.5)  # Neutral reward on complete failure
    print("Language Reward", rewards)
    return rewards

