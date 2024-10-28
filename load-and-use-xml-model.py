# code to know info about the xml model

# Model analysis code
# from openvino.runtime import Core

# ie = Core()
# model = ie.read_model(model_xml)

# print("\nModel Analysis:")
# print("Input layers:")
# for input_layer in model.inputs:
#     print(f"- Name: {input_layer.get_any_name()}")
#     print(f"- Shape: {input_layer.partial_shape}")
#     print(f"- Type: {input_layer.element_type}")

# print("\nOutput layers:")
# for output_layer in model.outputs:
#     print(f"- Name: {output_layer.get_any_name()}")
#     print(f"- Shape: {output_layer.partial_shape}")
#     print( f"- Type: {output_layer.element_type}")



# # First, let's analyze the decoder model structure
# ie = Core()
# decoder_model = ie.read_model(model=decoder_xml, weights=decoder_bin)

# print("\nDecoder Model Analysis:")
# print("Input layers:")
# for input_layer in decoder_model.inputs:
#     print(f"- Name: {input_layer.get_any_name()}")
#     print(f"- Shape: {input_layer.partial_shape}")
#     print(f"- Type: {input_layer.element_type}")

# print("\nOutput layers:")
# for output_layer in decoder_model.outputs:
#     print(f"- Name: {output_layer.get_any_name()}")
#     print(f"- Shape: {output_layer.partial_shape}")
#     print(f"- Type: {output_layer.element_type}")



from openvino.runtime import Core, PartialShape
import numpy as np
import librosa

# Define paths
encoder_xml = r"C:\\Users\\HP\\openvino_project\\output_folder\\encoder-asr_model.xml"
encoder_bin = r"C:\\Users\\HP\\openvino_project\\output_folder\\encoder-asr_model.bin"
decoder_xml = r"C:\\Users\\HP\\openvino_project\\output_folder_decoder\\decoder_joint-asr_model.xml"
decoder_bin = r"C:\\Users\\HP\\openvino_project\\output_folder_decoder\\decoder_joint-asr_model.bin"
audio_path = r"C:\Users\HP\OneDrive\Desktop\optmizedModel\Hamaspyur76ct_1025071435686_500.wav"

def create_char_map():
    char_map = {}
    # Armenian uppercase letters (0x0531-0x0556)
    offset = 0x0531
    for i in range(0, 38):  # Adjust range based on your model's output
        char_map[i] = chr(offset + i)
    
    # Add space and punctuation
    char_map[38] = " "
    char_map[256] = "<eos>"
    return char_map

char_map = create_char_map()

def preprocess_audio(audio_path, n_mels=80, target_sr=16000):
    audio_data, sr = librosa.load(audio_path, sr=target_sr)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=target_sr,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160,
        win_length=400
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    normalized = (log_mel - mean) / (std + 1e-5)
    normalized = normalized[np.newaxis, :, :]
    length = np.array([normalized.shape[2]], dtype=np.int64)
    return normalized.astype(np.float32), length

def decode_predictions(predictions, char_map):
    text = []
    prev_token = None
    for token in predictions:
        # Skip EOS and repeated tokens
        if token != 256 and token != prev_token:
            if token in char_map:
                text.append(char_map[token])
            else:
                text.append(f"<{token}>")  # Mark unknown tokens
        prev_token = token
    return ''.join(text)

try:
    # Initialize OpenVINO runtime
    ie = Core()
    
    # Load and compile encoder
    encoder_model = ie.read_model(model=encoder_xml, weights=encoder_bin)
    compiled_encoder = ie.compile_model(model=encoder_model, device_name="CPU")
    
    # Process audio through encoder
    input_data, length_data = preprocess_audio(audio_path)
    encoder_inputs = {
        "audio_signal": input_data,
        "length": length_data
    }
    encoder_outputs = compiled_encoder(encoder_inputs)
    encoder_features = encoder_outputs[compiled_encoder.output("outputs")]
    
    print(f"\nEncoder output shape: {encoder_features.shape}")
    
    # Load and compile decoder
    decoder_model = ie.read_model(model=decoder_xml, weights=decoder_bin)
    compiled_decoder = ie.compile_model(model=decoder_model, device_name="CPU")
    
    # Initialize states and inputs for decoder
    batch_size = encoder_features.shape[0]
    max_length = encoder_features.shape[2]
    hidden_size = 640
    
    # Store predictions
    all_predictions = []
    current_states_1 = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    current_states_2 = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    
    # Iterative decoding
    for i in range(max_length):
        decoder_inputs = {
            "encoder_outputs": encoder_features,
            "targets": np.array([[all_predictions[-1] if all_predictions else 0]], dtype=np.int32),
            "prednet_lengths": np.array([1], dtype=np.int32),
            "input_states_1": current_states_1,
            "input_states_2": current_states_2
        }
        
        decoder_outputs = compiled_decoder(decoder_inputs)
        output_logits = decoder_outputs[compiled_decoder.output("outputs")]
        current_states_1 = decoder_outputs[compiled_decoder.output("output_states_1")]
        current_states_2 = decoder_outputs[compiled_decoder.output("output_states_2")]
        
        # Get top K predictions
        probs = output_logits[0, 0, 0]
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        # Take the most likely non-EOS token
        for idx in top_indices:
            if idx != 256:  # If not EOS token
                pred = idx
                break
        else:
            pred = top_indices[0]  # If all are EOS, take the most likely
        
        # Break if we hit the EOS token
        if pred == 256:
            break
            
        all_predictions.append(int(pred))
    
    # Print transcription
    transcribed_text = decode_predictions(all_predictions, char_map)
    
    print("\nTranscribed Text:")
    print(transcribed_text)
    
    print("\nPredicted tokens:")
    print(all_predictions)
    
    # Print token to character mapping for debugging
    print("\nToken mapping for first few predictions:")
    for token in all_predictions[:10]:
        if token in char_map:
            print(f"Token {token} -> '{char_map[token]}'")
        else:
            print(f"Token {token} -> Unknown")

except Exception as e:
    print(f"\nError: {str(e)}")





# from openvino.runtime import Core, PartialShape
# import numpy as np
# import librosa

# # Define paths
# encoder_xml = r"C:\\Users\\HP\\openvino_project\\output_folder\\encoder-asr_model.xml"
# encoder_bin = r"C:\\Users\\HP\\openvino_project\\output_folder\\encoder-asr_model.bin"
# decoder_xml = r"C:\\Users\\HP\\openvino_project\\output_folder_decoder\\decoder_joint-asr_model.xml"
# decoder_bin = r"C:\\Users\\HP\\openvino_project\\output_folder_decoder\\decoder_joint-asr_model.bin"
# audio_path = r"C:\Users\HP\OneDrive\Desktop\optmizedModel\Hamaspyur76ct_1025071435686_500.wav"

# def create_armenian_char_map():
#     # Armenian Unicode range: 0x0530-0x058F
#     char_map = {}
#     # Add space
#     char_map[0] = " "
#     # Add Armenian letters
#     for i, code_point in enumerate(range(0x0531, 0x0557), start=1):  # Armenian uppercase
#         char_map[i] = chr(code_point)
#     for i, code_point in enumerate(range(0x0561, 0x0587), start=39):  # Armenian lowercase
#         char_map[i] = chr(code_point)
#     # Add special tokens
#     char_map[256] = "<eos>"
#     return char_map

# armenian_chars = create_armenian_char_map()

# def preprocess_audio(audio_path, n_mels=80, target_sr=16000):
#     audio_data, sr = librosa.load(audio_path, sr=target_sr)
#     mel_spec = librosa.feature.melspectrogram(
#         y=audio_data,
#         sr=target_sr,
#         n_mels=n_mels,
#         n_fft=400,
#         hop_length=160,
#         win_length=400
#     )
#     log_mel = librosa.power_to_db(mel_spec, ref=np.max)
#     mean = np.mean(log_mel)
#     std = np.std(log_mel)
#     normalized = (log_mel - mean) / (std + 1e-5)
#     normalized = normalized[np.newaxis, :, :]
#     length = np.array([normalized.shape[2]], dtype=np.int64)
#     return normalized.astype(np.float32), length

# def decode_predictions(predictions, char_map):
#     text = []
#     prev_token = None
#     for token in predictions:
#         if token == 256:  # EOS token
#             break
#         # Skip repeated tokens and convert to character
#         if token != prev_token and token in char_map:
#             text.append(char_map[token])
#         prev_token = token
#     return ''.join(text)

# try:
#     # Initialize OpenVINO runtime
#     ie = Core()
    
#     # Load and compile encoder
#     encoder_model = ie.read_model(model=encoder_xml, weights=encoder_bin)
#     compiled_encoder = ie.compile_model(model=encoder_model, device_name="CPU")
    
#     # Process audio through encoder
#     input_data, length_data = preprocess_audio(audio_path)
#     encoder_inputs = {
#         "audio_signal": input_data,
#         "length": length_data
#     }
#     encoder_outputs = compiled_encoder(encoder_inputs)
#     encoder_features = encoder_outputs[compiled_encoder.output("outputs")]
    
#     print(f"\nEncoder output shape: {encoder_features.shape}")
    
#     # Load and compile decoder
#     decoder_model = ie.read_model(model=decoder_xml, weights=decoder_bin)
#     compiled_decoder = ie.compile_model(model=decoder_model, device_name="CPU")
    
#     # Initialize states and inputs for decoder
#     batch_size = encoder_features.shape[0]
#     max_length = encoder_features.shape[2]
#     hidden_size = 640
    
#     # Store predictions
#     all_predictions = []
#     current_states_1 = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
#     current_states_2 = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    
#     # Iterative decoding
#     for i in range(max_length):
#         decoder_inputs = {
#             "encoder_outputs": encoder_features,
#             "targets": np.array([[all_predictions[-1] if all_predictions else 0]], dtype=np.int32),
#             "prednet_lengths": np.array([1], dtype=np.int32),
#             "input_states_1": current_states_1,
#             "input_states_2": current_states_2
#         }
        
#     decoder_outputs = compiled_decoder(decoder_inputs)
#     output_logits = decoder_outputs[compiled_decoder.output("outputs")]
    
#     # Look at all probable tokens, not just the top one
#     probabilities = output_logits[0, 0, 0]  # First position
    
#     # Get top 10 most probable tokens
#     top_k = 10
#     top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
#     print("\nTop", top_k, "most probable tokens:")
#     for idx in top_indices:
#         prob = probabilities[idx]
#         print(f"Token {idx}: probability {prob:.4f}")
#         if 0x0530 <= idx <= 0x058F:  # If in Armenian Unicode range
#             print(f"  - Armenian character: {chr(idx)}")
        
#     # Print unique values in the logits
#     unique_vals = np.unique(output_logits)
#     print("\nNumber of unique logit values:", len(unique_vals))
#     print("Range of logit values:", np.min(unique_vals), "to", np.max(unique_vals))

# except Exception as e:
#     print(f"\nError: {str(e)}")
