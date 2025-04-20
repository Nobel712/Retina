import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# 1) Define All Custom Layers
# -----------------------------

# Convolution Block
class ConvBlock(layers.Layer):
    def __init__(self, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

# Downsampling Block
class DownSample(layers.Layer):
    def __init__(self, pool_size=(2, 2)):
        super(DownSample, self).__init__()
        self.pool = layers.MaxPooling2D(pool_size)

    def call(self, x):
        return self.pool(x)

# Upsampling Block
class UpSample(layers.Layer):
    def __init__(self, out_channels):
        super(UpSample, self).__init__()
        self.conv1x1 = layers.Conv2D(out_channels, (1, 1), activation="relu")
        self.upsample = layers.UpSampling2D(size=(2, 2))

    def call(self, x):
        x = self.upsample(x)
        x = self.conv1x1(x)
        return x

# Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)

# GRU Model
class GRUModel(layers.Layer):
    def __init__(self, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.bi_gru = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True))
        self.fc = layers.Dense(output_size)

    def call(self, inputs):
        x = self.bi_gru(inputs)
        output = self.fc(x)
        return output

# Feature Map Fusion
class AdvancedDiffusionFeatureMapFusion(layers.Layer):
    def __init__(self, out_channels):
        super(AdvancedDiffusionFeatureMapFusion, self).__init__()
        
        # Learnable convolution to process the fusion of feature maps
        self.conv1x1 = layers.Conv2D(out_channels, (1, 1), activation="relu")
        
        # Diffusion-like filter: used to integrate features with spatial adaptation
        self.diffusion_filter = layers.Conv2D(out_channels, (3, 3), padding="same", activation="relu", use_bias=False)
        
        # Attention mechanism to focus on important features
        self.attention = layers.Attention()
        
        # A convolutional layer to process the attention-weighted fusion
        self.attention_weighted_conv = layers.Conv2D(out_channels, (1, 1), activation="relu")

    def call(self, encode_map, GRU_map, Decode_map):
        """
        Forward pass that integrates the feature maps using diffusion-like mechanisms.
        
        encode_map, GRU_map, Decode_map: These are the feature maps to be fused.
        """

        # Step 1: Resize GRU map to match the encoder's spatial dimensions
        GRU_map_resized = layers.Lambda(lambda x: tf.image.resize(x, size=(encode_map.shape[1], encode_map.shape[2])))(GRU_map)
        
        # Step 2: Concatenate feature maps for fusion
        fused_map = tf.concat([encode_map, GRU_map_resized, Decode_map], axis=-1)
        
        # Step 3: Apply diffusion filtering (spatial context integration)
        diffused_map = self.diffusion_filter(fused_map)
        
        # Step 4: Apply attention mechanism for adaptive fusion
        attention_weights = self.attention([diffused_map, diffused_map])  # Self-attention for feature adaptation
        
        # Step 5: Weighted fusion based on attention scores
        attention_fused_map = self.attention_weighted_conv(attention_weights * diffused_map)
        
        # Step 6: Final fusion through 1x1 convolution
        return self.conv1x1(attention_fused_map)




# 2) Build the Segmentation Model Architecture
def build_segmentation_model(input_shape=(256, 256, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # ----------------
    # Encoder
    # ----------------
    R1 = ConvBlock(64)(inputs)      # (256,256,64)
    P1 = DownSample()(R1)           # (128,128,64)

    R2 = ConvBlock(128)(P1)         # (128,128,128)
    P2 = DownSample()(R2)           # (64,64,128)

    R3 = ConvBlock(256)(P2)         # (64,64,256)
    P3 = DownSample()(R3)           # (32,32,256)

    R4 = ConvBlock(512)(P3)         # (32,32,512)
    P4 = DownSample()(R4)           # (16,16,512)

    # ----------------
    # Transformer Bottleneck
    # ----------------
    seq_len_p4 = 16 * 16
    transformer_input = layers.Reshape((seq_len_p4, 512))(P4)
    transformer_out = TransformerBlock(embed_dim=512, num_heads=4, ff_dim=1024)(transformer_input)
    transformer_out = layers.Reshape((16, 16, 512))(transformer_out)

    # ----------------
    # GRU Layers or Bi-GRU
    # ----------------
    G1 = GRUModel(128, 512)(layers.Reshape((16*16, 512))(transformer_out))
    G1 = layers.Reshape((16, 16, 512))(G1)

    G2 = GRUModel(128, 256)(layers.Reshape((32*32, 512))(R4))
    G2 = layers.Reshape((32, 32, 256))(G2)

    G3 = GRUModel(128, 128)(layers.Reshape((64*64, 256))(R3))
    G3 = layers.Reshape((64, 64, 128))(G3)

    G4 = GRUModel(128, 64)(layers.Reshape((128*128, 128))(R2))
    G4 = layers.Reshape((128, 128, 64))(G4)

    # ----------------
    # Decoder with Feature Fusion
    # ----------------
    fu1 = AdvancedDiffusionFeatureMapFusion(512)
    fu2 = AdvancedDiffusionFeatureMapFusion(256)
    fu3 = AdvancedDiffusionFeatureMapFusion(128)
    fu4 = AdvancedDiffusionFeatureMapFusion(64)

    U1 = UpSample(512)(G1)
    O1 = ConvBlock(512)(fu1(R4, U1, P3))

    U2 = UpSample(256)(O1)
    G2_up = layers.Lambda(lambda x: tf.image.resize(x, size=(64, 64)))(G2)
    O2 = ConvBlock(256)(fu2(R3, G2_up, U2))

    U3 = UpSample(128)(O2)
    G3_up = layers.Lambda(lambda x: tf.image.resize(x, size=(128, 128)))(G3)
    O3 = ConvBlock(128)(fu3(R2, G3_up, U3))

    U4 = UpSample(64)(O3)
    G4_up = layers.Lambda(lambda x: tf.image.resize(x, size=(256, 256)))(G4)
    O4 = ConvBlock(64)(fu4(R1, G4_up, U4))

    # ----------------
    # Output Segmentation Mask
    # ----------------
    outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(O4)

    return models.Model(inputs, outputs)

model = build_segmentation_model()
model.summary()
