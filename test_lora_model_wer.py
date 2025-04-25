#!/usr/bin/env python3
import os
import argparse
import torch
import pandas as pd
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation

def load_reference_texts(csv_path):
    """
    Load reference texts from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary mapping file_cut to reference text
    """
    try:
        df = pd.read_csv(csv_path)
        ref_texts = {}
        
        for _, row in df.iterrows():
            if 'file_cut' in row and 'transcriptions' in row:
                ref_texts[row['file_cut']] = row['transcriptions']
        
        print(f"Loaded {len(ref_texts)} reference texts from {csv_path}")
        return ref_texts
    except Exception as e:
        print(f"Error loading reference texts: {e}")
        return {}

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text (model transcription)
        
    Returns:
        WER as a float between 0 and 1
    """
    if not reference or not hypothesis:
        return 1.0  # Return 100% error if either is empty
    
    # Normalize texts (lowercase and remove punctuation)
    transform = Compose([ToLowerCase(), RemovePunctuation()])
    reference_norm = transform(reference)
    hypothesis_norm = transform(hypothesis)
    
    # Calculate WER
    return wer(reference_norm, hypothesis_norm)

def transcribe_audio(audio_path, base_model, adapter_path=None, language="en", task="transcribe"):
    """
    Transcribe an audio file using a Whisper model with optional LoRA adapter.
    
    Args:
        audio_path: Path to the audio file
        base_model: Path or identifier of the base Whisper model
        adapter_path: Path to the LoRA adapter (optional)
        language: Language code for transcription
        task: Task type ("transcribe" or "translate")
        
    Returns:
        Transcription text
    """
    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load processor and model
    print(f"Loading base model: {base_model}")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(base_model)
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
    
    # Load adapter if provided
    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        # Patch PeftModel.forward to handle num_items_in_batch
        try:
            _orig_forward = PeftModel.forward
            def _patched_forward(self, *args, **kwargs):
                kwargs.pop("num_items_in_batch", None)
                return _orig_forward(self, *args, **kwargs)
            PeftModel.forward = _patched_forward
        except:
            # Handle case where patch is already applied
            pass
            
        whisper_model = PeftModel.from_pretrained(whisper_model, adapter_path)
        print("LoRA adapter loaded successfully")
    
    whisper_model = whisper_model.to(device)
    
    # Load audio
    print(f"Loading audio file: {audio_path}")
    audio, sampling_rate = sf.read(audio_path)
    print(f"Audio length: {len(audio)/sampling_rate:.2f} seconds, Sample rate: {sampling_rate}Hz")
    
    # Process input features
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def main():
    parser = argparse.ArgumentParser(description="Test a Whisper model with optional LoRA adapter")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="openai/whisper-small", 
                        help="Base model name or path (default: openai/whisper-small)")
    parser.add_argument("--adapter_path", type=str, default=None, 
                        help="Path to the LoRA adapter (optional)")
    
    # Audio parameters
    parser.add_argument("--audio_path", type=str, required=True, 
                        help="Path to the audio file to transcribe")
    parser.add_argument("--language", type=str, default="en", 
                        help="Language code (default: en)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task type (default: transcribe)")
    
    # Batch processing
    parser.add_argument("--batch_mode", action="store_true", 
                        help="Process multiple audio files from a directory")
    parser.add_argument("--output_file", type=str, default="transcriptions.txt",
                        help="Output file for batch transcriptions (default: transcriptions.txt)")
    
    # WER calculation
    parser.add_argument("--reference_csv", type=str, default=None,
                        help="Path to CSV file containing reference texts")
    
    args = parser.parse_args()
    
    # Load reference texts if provided
    reference_texts = {}
    if args.reference_csv:
        reference_texts = load_reference_texts(args.reference_csv)
    
    # Batch mode: process multiple files
    if args.batch_mode:
        if not os.path.isdir(args.audio_path):
            print(f"Error: {args.audio_path} is not a directory. For batch mode, provide a directory path.")
            return
            
        audio_files = []
        for root, _, files in os.walk(args.audio_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"No audio files found in {args.audio_path}")
            return
            
        print(f"Found {len(audio_files)} audio files to process")
        
        # Results storage
        results = []
        total_wer = 0
        files_with_ref = 0
        
        with open(args.output_file, 'w') as f:
            f.write("Audio File | Reference | Transcription | WER\n")
            f.write("-" * 100 + "\n")
            
            for audio_file in audio_files:
                try:
                    transcription = transcribe_audio(
                        audio_file, 
                        args.base_model, 
                        args.adapter_path, 
                        args.language, 
                        args.task
                    )
                    
                    # Get file_cut from path
                    file_name = os.path.basename(audio_file)
                    rel_path = os.path.relpath(audio_file, args.audio_path)
                    
                    # Calculate WER if reference text is available
                    reference = ""
                    file_wer = -1  # -1 means no reference
                    
                    if reference_texts:
                        # Try to match the file name pattern in reference texts
                        for ref_file, ref_text in reference_texts.items():
                            if file_name == ref_file or file_name.startswith(os.path.splitext(ref_file)[0]):
                                reference = ref_text
                                file_wer = calculate_wer(reference, transcription)
                                total_wer += file_wer
                                files_with_ref += 1
                                break
                    
                    # Store and output result
                    results.append((rel_path, reference, transcription, file_wer))
                    
                    if file_wer >= 0:
                        f.write(f"{rel_path} | {reference} | {transcription} | {file_wer:.4f}\n")
                        print(f"Processed: {rel_path}")
                        print(f"Reference: {reference}")
                        print(f"Transcription: {transcription}")
                        print(f"WER: {file_wer:.4f}")
                    else:
                        f.write(f"{rel_path} | N/A | {transcription} | N/A\n")
                        print(f"Processed: {rel_path}")
                        print(f"Transcription: {transcription}")
                        print("No reference text available for WER calculation")
                    
                    print("-" * 40)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
            
            # Write overall WER if references were available
            if files_with_ref > 0:
                avg_wer = total_wer / files_with_ref
                f.write("\n" + "-" * 100 + "\n")
                f.write(f"Overall WER: {avg_wer:.4f} (calculated on {files_with_ref} files)\n")
                print(f"Overall WER: {avg_wer:.4f} (calculated on {files_with_ref} files)")
        
        print(f"All transcriptions saved to {args.output_file}")
    
    # Single file mode
    else:
        try:
            transcription = transcribe_audio(
                args.audio_path, 
                args.base_model, 
                args.adapter_path, 
                args.language, 
                args.task
            )
            
            # Get file_cut from path
            file_name = os.path.basename(args.audio_path)
            
            # Calculate WER if reference text is available
            reference = ""
            file_wer = -1  # -1 means no reference
            
            if reference_texts:
                # Try to match the file name pattern in reference texts
                for ref_file, ref_text in reference_texts.items():
                    if file_name == ref_file or file_name.startswith(os.path.splitext(ref_file)[0]):
                        reference = ref_text
                        file_wer = calculate_wer(reference, transcription)
                        break
            
            print("\nTranscription result:")
            print("-" * 40)
            if reference:
                print(f"Reference: {reference}")
            print(f"Transcription: {transcription}")
            if file_wer >= 0:
                print(f"WER: {file_wer:.4f}")
            elif reference_texts:
                print("No matching reference text found for WER calculation")
            print("-" * 40)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()