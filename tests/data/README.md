# Test Data

This directory contains test models and fixtures.

To run comprehensive tests with real models:
1. Export a YOLOv8n model to TFLite INT8 format
2. Place it here as `yolov8n_int8.tflite`
3. Or set OPENEDGE_TEST_MODEL environment variable to model path

Without a real model, TensorFlow-dependent tests will be skipped.
