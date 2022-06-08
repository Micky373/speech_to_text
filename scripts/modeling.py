import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
import wave
import mlflow
from jiwer import wer

# An integer scalar Tensor. The window length in samples.
frame_length = 256 #this should be less than or equal to fft length
# An integer scalar Tensor. The number of samples to step.
frame_step = STEP
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = MFCC_DIME
# The set of characters accepted in the transcription.
characters = """
ሀ ሁ ሂ ሄ ህ ሆ
ለ ሉ ሊ ላ ሌ ል ሎ ሏ
መ ሙ ሚ ማ ሜ ም ሞ ሟ
ረ ሩ ሪ ራ ሬ ር ሮ ሯ
ሰ ሱ ሲ ሳ ሴ ስ ሶ ሷ
ሸ ሹ ሺ ሻ ሼ ሽ ሾ ሿ
ቀ ቁ ቂ ቃ ቄ ቅ ቆ ቋ
በ ቡ ቢ ባ ቤ ብ ቦ ቧ
ቨ ቩ ቪ ቫ ቬ ቭ ቮ ቯ
ተ ቱ ቲ ታ ቴ ት ቶ ቷ
ቸ ቹ ቺ ቻ ቼ ች ቾ ቿ
ኋ
ነ ኑ ኒ ና ኔ ን ኖ ኗ
ኘ ኙ ኚ ኛ ኜ ኝ ኞ ኟ
አ ኡ ኢ ኤ እ ኦ
ኧ
ከ ኩ ኪ ካ ኬ ክ ኮ
ኳ
ወ ዉ ዊ ዋ ዌ ው ዎ
ዘ ዙ ዚ ዛ ዜ ዝ ዞ ዟ
ዠ ዡ ዢ ዣ ዤ ዥ ዦ ዧ
የ ዩ ዪ ያ ዬ ይ ዮ
ደ ዱ ዲ ዳ ዴ ድ ዶ ዷ
ጀ ጁ ጂ ጃ ጄ ጅ ጆ ጇ
ገ ጉ ጊ ጋ ጌ ግ ጐ ጓ ጔ
ጠ ጡ ጢ ጣ ጤ ጥ ጦ ጧ
ጨ ጩ ጪ ጫ ጬ ጭ ጮ ጯ
ጰ ጱ ጲ ጳ ጴ ጵ ጶ ጷ
ፀ ፁ ፂ ፃ ፄ ፅ ፆ ፇ
ፈ ፉ ፊ ፋ ፌ ፍ ፎ ፏ
ፐ ፑ ፒ ፓ ፔ ፕ ፖ
""".replace('\n',' ').split(' ')
characters = characters[:-1]
characters.insert(1, ' ')

# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

class Modeling():
    """
    Building Model
    training the model
    Evaluating the model performance
    """
    
    def __init__(self) -> None:
        pass
        
    def encode_single_sample(self, wav_file, label):
        """
        Process the Audio
        
        """
        #read wav file
        file = tf.io.read_file(wav_file)
        #decode voice file
        audio, _ =tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        #change type to float32
        audio = tf.cast(audio, tf.float32)    
        #get the spectrogram
        spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
        #we only need the magnitude of the spectrogram
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        #normalization
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs= tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        """
        Process the label
        """
        #convert label to lower case
        label = tf.strings.lower(label)
        #split label
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        # Map the characters in label to numbers
        label = char_to_num(label)
        #  Return a dict as our model is expecting two inputs
        return spectrogram, label

    def creating_dataset_object(self, train_meta, valid_meta, BS):
        batch_size = BS
        # Define the trainig dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (list(train_meta["Feature"]), list(train_meta["Target"]))
        )
        train_dataset = (
            train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # Define the validation dataset
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (list(valid_meta["Feature"]), list(valid_meta["Target"]))
        )
        validation_dataset = (
            validation_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return True
    #defining CTC loss function
    def CTCLoss(self, y_true, y_pred):
        #compute the training-time loss value 
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss
    
        #define our model
    def build_model(self, input_dim, output_dim, rnn_layers = 5, rnn_units =128):
        #Model input
        input_spectrogram = keras.layers.Input(shape=(None, input_dim), name="input")
        #Expand the dimensions to use 2D CNN
        x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
        #convolutions layer 1
        x = layers.Conv2D(filters=32, 
                        kernel_size=(11, 41), 
                        padding="same", 
                        strides = [2,2],
                        activation="relu", 
                        name="conv_1")(x)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        #convolution layer 2
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            activation = "relu",
            use_bias=False,
            name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        # Classification layer
        output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
        # Model
        model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model.compile(optimizer=opt, loss=self.CTCLoss)
        return model
    
    # A utility function to decode the output of the network
def decode_batch_predictions(self,pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)


    
