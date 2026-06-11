package com.utake.skufind;

import androidx.camera.core.ImageProxy;

import java.util.List;

public interface ProductAnalyzer {
    List<ProductDetection> analyze(ImageProxy image);
}
