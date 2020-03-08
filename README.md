# nnstreamer-gstreamer
merge gstreamer and nnstreamer

cmd : g++ nnstreamer_example_object_detection_tf.cc -o nntest2 `pkg-config --cflags --libs gstreamer-1.0 gstreamer-video-1.0 gtk+-3.0 gstreamer-app-1.0`
