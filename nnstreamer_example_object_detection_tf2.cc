/**
 * @file	nnstreamer_example_object_detection_tf.cc
 * @date	8 Jan 2019
 * @brief	Tensor stream example with Tensorflow model for object detection
 * @author	HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug		No known bugs.
 *
 * Get model by
 * $ cd $NNST_ROOT/bin
 * $ bash get-model.sh object-detection-tf
 * 
 * Run example :
 * Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plug-in.
 * $ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
 * $ ./nnstreamer_example_object_detection_tf
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <glib.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h> //@dw 20.03.08
#include <stdint.h> //@dw 20.03.08

#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <math.h>
#include <cairo.h>
#include <cairo-gobject.h>

// @yh 20.03.09 As you can see, this below code is made by DH.kim
int want = 1;
//double frame[640][64]={0,}; //@dw 20.03.09
double frame[630][64]={0,}; //@dw 20.03.09
//double frame2[1024][640]={0,};

static void prepare_buffer(GstAppSrc* appsrc, int index) {

  //static gboolean white = FALSE;
  static GstClockTime timestamp = 0;
  GstBuffer *buffer;
  guint size;
  GstFlowReturn ret;


  ////////////////////////////////////

  char filename[16];
  char s2[13]="data_point/";
  
  if(10>index){
    sprintf(filename,"000000000%d.txt",index);
  }else if(99>= index && index>=10){
    sprintf(filename,"00000000%d.txt",index);
  }else if(999>= index && index>=100){
    sprintf(filename,"0000000%d.txt",index);
  }
  
  
  char *filename2=strcat(s2, filename);
  printf("%s\n", filename2);

  FILE* fp;
  fp = fopen(filename2,"r");
  double read_num, x, y, z;
  double xy_theta, z_theta, range;
  double val = 57.2958;

  int cnt=1;

  while(fgetc(fp) != EOF){

    fscanf(fp, "%lf", &read_num);

    if(cnt%4==0){
        //calculate
        xy_theta=atan(y/x)*val+45;
        z_theta=round(atan(z/x)*val+5);
        if( 0<z_theta & z_theta<=64 & 0< xy_theta & xy_theta<=90 ){
               
            range=round(sqrt(x*x+y*y+z*z)*100)/100;
            xy_theta=round(xy_theta*7);
            //xy_theta_round=round(xy_theta);
            //y_theta_round=round(z_theta);
               
            frame[(int)xy_theta][(int)z_theta]=range;       
            //printf("range=%f  z_theta=%f  xy_theta=%f \n", range, z_theta , xy_theta);//frame=%f\n , frame[(int)xy_theta][(int)z_theta]             
        }

      /**
        x=(round(x)+320);
        y=(round(y)+612);
        if(x<640 & y<1024){
          frame2[(int)x][(int)y]=z;
          printf("zz=%f",z);
        }
      */
               
    }else if(cnt%4==1){
        //x
        x=read_num;
        //printf("x=%f",x);
    }else if(cnt%4==2){
        //y
        y=read_num;
        //printf(" y=%f",y);
    }else if(cnt%4==3){
        //z
        z=read_num;
        //printf(" z=%f\n",z);
    }
    cnt=cnt+1;
  }

  printf("number of point cnt=%d\n",cnt/4); 

  fclose(fp);
  

  /** 
  int col = 640;
  int row = 64;
  int cnt2=0;
  for(int i=0; i<col; i++){
      for(int j=0; j<row; j++){ 
            if(frame[i][j]>0){
                 //printf("%5.2f ",frame[i][j]);
                 cnt2=cnt2+1;
            }
      }
      //printf("\n");
  }
  printf("number of point=%d\n",cnt2);
  */ 
  ////////////////////////////////////


  if (!want) return;
  want = 0;
 

  /////////////////////
  size = 630*64*2; 
  buffer = gst_buffer_new_wrapped_full( (GstMemoryFlags)0, (gpointer)frame, size, 0, size, NULL, NULL );
  ////////////////////////


  GST_BUFFER_PTS (buffer) = timestamp;
  GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 4);

  timestamp += GST_BUFFER_DURATION (buffer);

  ret = gst_app_src_push_buffer(appsrc, buffer);

  if (ret != GST_FLOW_OK) {
    /* something wrong, stop pushing */
    // g_main_loop_quit (loop);
  }
}

static void cb_need_data (GstElement *appsrc, guint unused_size, gpointer user_data) {
  //prepare_buffer((GstAppSrc*)appsrc);
  want = 1;
}

// @dw 20.03.08


/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

/**
 * @brief Macro to check error case.
 */
// #define _check_cond_err(cond) \
//   do { \
//     if (!(cond)) { \
//       _print_log ("app failed! [line : %d]", __LINE__); \
//       goto error; \
//     } \
//   } while (0)

#define VIDEO_WIDTH     640
#define VIDEO_HEIGHT    480

#define BOX_SIZE        4
#define LABEL_SIZE      91
#define DETECTION_MAX   100

/**
 * @brief Max objects in display.
 */
#define MAX_OBJECT_DETECTION 5

typedef struct
{
  gint x;
  gint y;
  gint width;
  gint height;
  gint class_id;
  gfloat prob;
} DetectedObject;

typedef struct
{
  gboolean valid;
  GstVideoInfo vinfo;
} CairoOverlayState;

/**
 * @brief Data structure for tf model info.
 */
typedef struct
{
  gchar *model_path; /**< tf model file path */
  gchar *label_path; /**< label file path */
  GList *labels; /**< list of loaded labels */
} TFModelInfo;

/**
 * @brief Data structure for app.
 */
typedef struct
{
  GMainLoop *loop; /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for data stream */
  GstBus *bus; /**< gst bus for data pipeline */
  gboolean running; /**< true when app is running */
  GMutex mutex; /**< mutex for processing */
  TFModelInfo tf_info; /**< tf model info */
  CairoOverlayState overlay_state;
  std::vector<DetectedObject> detected_objects;
} AppData;

/**
 * @brief Data for pipeline and result.
 */
static AppData g_app;

/**
 * @brief Read strings from file.
 */
static gboolean
read_lines (const gchar * file_name, GList ** lines)
{
  std::ifstream file (file_name);
  if (!file) {
    _print_log ("Failed to open file %s", file_name);
    return FALSE;
  }

  std::string str;
  while (std::getline (file, str)) {
    *lines = g_list_append (*lines, g_strdup (str.c_str ()));
  }

  return TRUE;
}

/**
 * @brief Load labels.
 */
static gboolean
tf_load_labels (TFModelInfo * tf_info)
{
  g_return_val_if_fail (tf_info != NULL, FALSE);

  return read_lines (tf_info->label_path, &tf_info->labels);
}

/**
 * @brief Check tf model and load labels.
 */
static gboolean
tf_init_info (TFModelInfo * tf_info, const gchar * path)
{
  const gchar tf_model[] = "ssdlite_mobilenet_v2.pb";
  const gchar tf_label[] = "coco_labels_list.txt";

  g_return_val_if_fail (tf_info != NULL, FALSE);

  tf_info->model_path = g_strdup_printf ("%s/%s", path, tf_model);
  tf_info->label_path = g_strdup_printf ("%s/%s", path, tf_label);

  tf_info->labels = NULL;

  if (!g_file_test (tf_info->model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("cannot find tf model [%s]", tf_info->model_path);
    return FALSE;
  }
  if (!g_file_test (tf_info->label_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("cannot find tf label [%s]", tf_info->label_path);
    return FALSE;
  }

  g_return_val_if_fail (tf_load_labels (tf_info), FALSE);

  return TRUE;
}

/**
 * @brief Free data in tf info structure.
 */
static void
tf_free_info (TFModelInfo * tf_info)
{
  g_return_if_fail (tf_info != NULL);

  if (tf_info->model_path) {
    g_free (tf_info->model_path);
    tf_info->model_path = NULL;
  }

  if (tf_info->label_path) {
    g_free (tf_info->label_path);
    tf_info->label_path = NULL;
  }

  if (tf_info->labels) {
    g_list_free_full (tf_info->labels, g_free);
    tf_info->labels = NULL;
  }
}

/**
 * @brief Free resources in app data.
 */
static void
free_app_data (void)
{
  if (g_app.loop) {
    g_main_loop_unref (g_app.loop);
    g_app.loop = NULL;
  }

  if (g_app.bus) {
    gst_bus_remove_signal_watch (g_app.bus);
    gst_object_unref (g_app.bus);
    g_app.bus = NULL;
  }

  if (g_app.pipeline) {
    gst_object_unref (g_app.pipeline);
    g_app.pipeline = NULL;
  }

  g_app.detected_objects.clear ();

  tf_free_info (&g_app.tf_info);
  g_mutex_clear (&g_app.mutex);
}

/**
 * @brief Function to print error message.
 */
static void
parse_err_message (GstMessage * message)
{
  gchar *debug;
  GError *error;

  g_return_if_fail (message != NULL);

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error (message, &error, &debug);
      break;

    case GST_MESSAGE_WARNING:
      gst_message_parse_warning (message, &error, &debug);
      break;

    default:
      return;
  }

  gst_object_default_error (GST_MESSAGE_SRC (message), error, debug);
  g_error_free (error);
  g_free (debug);
}

/**
 * @brief Function to print qos message.
 */
static void
parse_qos_message (GstMessage * message)
{
  GstFormat format;
  guint64 processed;
  guint64 dropped;

  gst_message_parse_qos_stats (message, &format, &processed, &dropped);
  _print_log ("format[%d] processed[%" G_GUINT64_FORMAT "] dropped[%"
      G_GUINT64_FORMAT "]", format, processed, dropped);
}

/**
 * @brief Callback for message.
 */
static void
bus_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      _print_log ("received eos message");
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("received error message");
      parse_err_message (message);
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_WARNING:
      _print_log ("received warning message");
      parse_err_message (message);
      break;

    case GST_MESSAGE_STREAM_START:
      _print_log ("received start message");
      break;

    case GST_MESSAGE_QOS:
      parse_qos_message (message);
      break;

    default:
      break;
  }
}

/**
 * @brief Get detected objects.
 */
static void
get_detected_objects (
  gfloat * num_detections,
  gfloat * detection_classes,
  gfloat * detection_scores,
  gfloat * detection_boxes)
{

  g_mutex_lock (&g_app.mutex);

  g_app.detected_objects.clear ();

  _print_log("========================================================");
  _print_log("                 Number Of Objects: %2d", (int) num_detections[0]);
  _print_log("========================================================");
  for (int i = 0; i < (int) num_detections[0]; i++){
    DetectedObject object;

    object.class_id = (int) detection_classes[i];
    object.x = (int) (detection_boxes[i * BOX_SIZE + 1] * VIDEO_WIDTH);
    object.y = (int) (detection_boxes[i * BOX_SIZE] * VIDEO_HEIGHT);
    object.width = (int) ((detection_boxes[i * BOX_SIZE + 3]
                 - detection_boxes[i * BOX_SIZE + 1]) * VIDEO_WIDTH);
    object.height = (int) ((detection_boxes[i * BOX_SIZE + 2]
                  - detection_boxes[i * BOX_SIZE]) * VIDEO_HEIGHT);
    object.prob = detection_scores[i];

    _print_log("%10s: x:%3d, y:%3d, w:%3d, h:%3d, prob:%.2f",
      (gchar *) g_list_nth_data (g_app.tf_info.labels, object.class_id),
      object.x, object.y, object.width, object.height, object.prob);

    g_app.detected_objects.push_back (object);
  }
  _print_log("========================================================");

  g_mutex_unlock (&g_app.mutex);
}

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  GstMemory *mem_num, *mem_classes, *mem_scores, *mem_boxes;
  GstMapInfo info_num, info_classes, info_scores, info_boxes;
  gfloat *num_detections, *detection_classes, *detection_scores, *detection_boxes;

  g_return_if_fail (g_app.running);

  /**
   * tensor type is float32.
   * [0] dim of num_detections    > 1
   * [1] dim of detection_classes > 1: 100
   * [2] dim of detection_scores  > 1: 100
   * [3] dim of detection_boxes   > 1: 100: 4 (top, left, bottom, right)
   */
  g_assert (gst_buffer_n_memory (buffer) == 4);

  /* num_detections */
  mem_num = gst_buffer_get_memory (buffer, 0);
  g_assert (gst_memory_map (mem_num, &info_num, GST_MAP_READ));
  g_assert (info_num.size == 4);
  num_detections = (gfloat *) info_num.data;

  /* detection_classes */
  mem_classes = gst_buffer_get_memory (buffer, 1);
  g_assert (gst_memory_map (mem_classes, &info_classes, GST_MAP_READ));
  g_assert (info_classes.size == DETECTION_MAX * 4);
  detection_classes = (gfloat *) info_classes.data;

  /* detection_scores */
  mem_scores = gst_buffer_get_memory (buffer, 2);
  g_assert (gst_memory_map (mem_scores, &info_scores, GST_MAP_READ));
  g_assert (info_scores.size == DETECTION_MAX * 4);
  detection_scores = (gfloat *) info_scores.data;

  /* detection_boxes */
  mem_boxes = gst_buffer_get_memory (buffer, 3);
  g_assert (gst_memory_map (mem_boxes, &info_boxes, GST_MAP_READ));
  g_assert (info_boxes.size == DETECTION_MAX * BOX_SIZE * 4);
  detection_boxes = (gfloat *) info_boxes.data;

  get_detected_objects (
    num_detections, detection_classes, detection_scores, detection_boxes);

  gst_memory_unmap (mem_num, &info_num);
  gst_memory_unmap (mem_classes, &info_classes);
  gst_memory_unmap (mem_scores, &info_scores);
  gst_memory_unmap (mem_boxes, &info_boxes);

  gst_memory_unref (mem_num);
  gst_memory_unref (mem_classes);
  gst_memory_unref (mem_scores);
  gst_memory_unref (mem_boxes);
}

/**
 * @brief Set window title.
 * @param name GstXImageSink element name
 * @param title window title
 */
static void
set_window_title (const gchar * name, const gchar * title)
{
  GstTagList *tags;
  GstPad *sink_pad;
  GstElement *element;

  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), name);

  g_return_if_fail (element != NULL);

  sink_pad = gst_element_get_static_pad (element, "sink");

  if (sink_pad) {
    tags = gst_tag_list_new (GST_TAG_TITLE, title, NULL);
    gst_pad_send_event (sink_pad, gst_event_new_tag (tags));
    gst_object_unref (sink_pad);
  }

  gst_object_unref (element);
}

/**
 * @brief Store the information from the caps that we are interested in.
 */
static void
prepare_overlay_cb (GstElement * overlay, GstCaps * caps, gpointer user_data)
{
  CairoOverlayState *state = &g_app.overlay_state;

  state->valid = gst_video_info_from_caps (&state->vinfo, caps);
}

/**
 * @brief Callback to draw the overlay.
 */
static void
draw_overlay_cb (GstElement * overlay, cairo_t * cr, guint64 timestamp,
    guint64 duration, gpointer user_data)
{
  CairoOverlayState *state = &g_app.overlay_state;
  gfloat x, y, width, height;
  gchar *label;
  guint drawed = 0;

  g_return_if_fail (state->valid);
  g_return_if_fail (g_app.running);

  std::vector<DetectedObject> detected;
  std::vector<DetectedObject>::iterator iter;

  g_mutex_lock (&g_app.mutex);
  detected = g_app.detected_objects;
  g_mutex_unlock (&g_app.mutex);

  /* set font props */
  cairo_select_font_face (cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
      CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size (cr, 20.0);

  for (iter = detected.begin (); iter != detected.end (); ++iter) {
    label =
        (gchar *) g_list_nth_data (g_app.tf_info.labels, iter->class_id);

    x = iter->x;
    y = iter->y;
    width = iter->width;
    height = iter->height;

    /* draw rectangle */
    cairo_rectangle (cr, x, y, width, height);
    cairo_set_source_rgb (cr, 1, 0, 0);
    cairo_set_line_width (cr, 1.5);
    cairo_stroke (cr);
    cairo_fill_preserve (cr);

    /* draw title */
    cairo_move_to (cr, x + 5, y + 25);
    cairo_text_path (cr, label);
    cairo_set_source_rgb (cr, 1, 0, 0);
    cairo_fill_preserve (cr);
    cairo_set_source_rgb (cr, 1, 1, 1);
    cairo_set_line_width (cr, .3);
    cairo_stroke (cr);
    cairo_fill_preserve (cr);

    if (++drawed >= MAX_OBJECT_DETECTION) {
      /* max objects drawed */
      break;
    }
  }
}

/**
 * @brief Main function.
 */
int
main (int argc, char ** argv)
{

  
  //char *file_list[11]={"00","01","02","03","04","05","06","07","08","09","10"}; //@dw 20.03.08
  /////
  int file_list[157]={0,};
  for(int i=0; i<157;i++){
      file_list[i]=i;
      //printf("ss==%ld",sizeof(file_list));
  }
  ////

  const gchar tf_model_path[] = "./tf_model";

  gchar *str_pipeline;
  //GstElement *element;
  GstElement *element, *pipeline, *appsrc, *conv, *videosink; //@dw 20.03.08
  
  _print_log ("start app..");

  #if 0
  /* init app variable */
  g_app.running = FALSE;
  g_app.loop = NULL;
  g_app.bus = NULL;
  g_app.pipeline = NULL;
  g_app.detected_objects.clear ();
  g_mutex_init (&g_app.mutex);
  #endif

 // _check_cond_err (tf_init_info (&g_app.tf_info, tf_model_path));

  /* init gstreamer */
  gst_init (&argc, &argv);

    #if 0 //used for tf model @dw 20.03.08
  /* main loop */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  /* init pipeline */

   str_pipeline =
      g_strdup_printf
      ("v4l2src name=src ! videoconvert ! videoscale ! video/x-raw,width=%d,height=%d,format=RGB ! tee name=t_raw "
      "t_raw. ! queue ! videoconvert ! cairooverlay name=tensor_res ! ximagesink name=img_tensor "
      "t_raw. ! queue leaky=2 max-size-buffers=2 ! videoscale ! tensor_converter ! "
      "tensor_filter framework=tensorflow model=%s "
      "input=3:640:480:1 inputname=image_tensor inputtype=uint8 "
      "output=1,100:1,100:1,4:100:1 "
      "outputname=num_detections,detection_classes,detection_scores,detection_boxes "
      "outputtype=float32,float32,float32,float32 ! "
      "tensor_sink name=tensor_sink ",
      VIDEO_WIDTH, VIDEO_HEIGHT, g_app.tf_info.model_path);
  #endif

  // setup pipeline
  pipeline = gst_pipeline_new ("pipeline");
  appsrc = gst_element_factory_make ("appsrc", "source");
  conv = gst_element_factory_make ("videoconvert", "conv");
  videosink = gst_element_factory_make ("xvimagesink", "videosink");

/* setup */
  g_object_set (G_OBJECT (appsrc), "caps",
      gst_caps_new_simple ("video/x-raw",
             "format", G_TYPE_STRING, "RGB16",
             "width", G_TYPE_INT, 630, //@dw 20.03.09
             "height", G_TYPE_INT, 64,
             "framerate", GST_TYPE_FRACTION, 25, 1,
             NULL), NULL);
  gst_bin_add_many (GST_BIN (pipeline), appsrc, conv, videosink, NULL);
  gst_element_link_many (appsrc, conv, videosink, NULL);

  /* setup appsrc */
  g_object_set (G_OBJECT (appsrc),
    "stream-type", 0, // GST_APP_STREAM_TYPE_STREAM
    "format", GST_FORMAT_TIME,
        "is-live", TRUE,
    NULL);
  g_signal_connect (appsrc, "need-data", G_CALLBACK (cb_need_data), NULL);

  /* play */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);


  for(int i=0; i<sizeof(file_list)/8; i++) {
    prepare_buffer((GstAppSrc*)appsrc, file_list[i]);
    g_main_context_iteration(g_main_context_default(),FALSE);
  }

    /* clean up */
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (GST_OBJECT (pipeline));



  #if 0 //used for tf model @dw 20.03.08
  /* setup */
  g_object_set (G_OBJECT (appsrc), "caps",
      gst_caps_new_simple ("video/x-raw",
             "format", G_TYPE_STRING, "RGB16",
             "width", G_TYPE_INT, 640,
             "height", G_TYPE_INT, 64,
             "framerate", GST_TYPE_FRACTION, 25, 1,
             NULL), NULL);
  gst_bin_add_many (GST_BIN (pipeline), appsrc, conv, videosink, NULL);
  gst_element_link_many (appsrc, conv, videosink, NULL);

  /* setup appsrc */
  g_object_set (G_OBJECT (appsrc),
    "stream-type", 0, // GST_APP_STREAM_TYPE_STREAM
    "format", GST_FORMAT_TIME,
        "is-live", TRUE,
    NULL);
  g_signal_connect (appsrc, "need-data", G_CALLBACK (cb_need_data), NULL);
  #endif


  //_print_log ("%s\n", str_pipeline);

  #if 0 

  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.pipeline != NULL);

  /* bus and message callback */
  g_app.bus = gst_element_get_bus (g_app.pipeline);
  _check_cond_err (g_app.bus != NULL);

  gst_bus_add_signal_watch (g_app.bus);
  g_signal_connect (g_app.bus, "message", G_CALLBACK (bus_message_cb), NULL);

  /* tensor sink signal : new data callback */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_sink");
  g_signal_connect (element, "new-data", G_CALLBACK (new_data_cb), NULL);
  gst_object_unref (element);

  /* cairo overlay */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_res");
  g_signal_connect (element, "draw", G_CALLBACK (draw_overlay_cb), NULL);
  g_signal_connect (element, "caps-changed", G_CALLBACK (prepare_overlay_cb),
      NULL);
  gst_object_unref (element);

  /* start pipeline */
  gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);
  g_app.running = TRUE;

  /* set window title */
  set_window_title ("img_tensor", "NNStreamer Example");

  /* run main loop */
  g_main_loop_run (g_app.loop);

  /* quit when received eos or error message */
  g_app.running = FALSE;

  /* cam source element */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "src");

  gst_element_set_state (element, GST_STATE_READY);
  gst_element_set_state (g_app.pipeline, GST_STATE_READY);

  g_usleep (200 * 1000);

  gst_element_set_state (element, GST_STATE_NULL);
  gst_element_set_state (g_app.pipeline, GST_STATE_NULL);

  g_usleep (200 * 1000);
  gst_object_unref (element);

error:
  _print_log ("close app..");

  free_app_data ();
  #endif 

  return 0;
}