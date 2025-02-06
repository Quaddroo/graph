#version 430

layout(local_size_x = 256) in;

// Input data buffers
layout(std430, binding = 0) buffer XData {
    float x[];
};
layout(std430, binding = 1) buffer YData {
    float y[];
};

// Output buffer for OHLC data
layout(std430, binding = 2) buffer OHLCData {
    float ohlc[];
};

uniform int n;  // Resampling chunk size

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint idx = gid * n;

    // Bounds check
    if (idx >= y.length()) return;

    // Compute OHLC
    float open_ = y[idx];
    float high = open_;
    float low = open_;
    uint lastIndex = min(idx + n - 1, y.length() - 1);
    float close = y[lastIndex];

    for (uint i = idx; i <= lastIndex; ++i) {
        high = max(high, y[i]);
        low = min(low, y[i]);
    }

    // Store results in the output buffer
    uint output_idx = gid * 4;
    ohlc[output_idx + 0] = open_;
    ohlc[output_idx + 1] = high;
    ohlc[output_idx + 2] = low;
    ohlc[output_idx + 3] = close;
}
