#!/usr/bin/env python3
import os
import argparse
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

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
    
    args = parser.parse_args()
    
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
        
        with open(args.output_file, 'w') as f:
            f.write("Audio File | Transcription\n")
            f.write("-" * 80 + "\n")
            
            for audio_file in audio_files:
                try:
                    transcription = transcribe_audio(
                        audio_file, 
                        args.base_model, 
                        args.adapter_path, 
                        args.language, 
                        args.task
                    )
                    
                    rel_path = os.path.relpath(audio_file, args.audio_path)
                    f.write(f"{rel_path} | {transcription}\n")
                    print(f"Processed: {rel_path}")
                    print(f"Transcription: {transcription}")
                    print("-" * 40)
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
        
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
            
            print("\nTranscription result:")
            print("-" * 40)
            print(transcription)
            print("-" * 40)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()