package aymanzeine.objectrecog;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class RealtimeActivity extends AppCompatActivity implements View.OnTouchListener, CameraBridgeViewBase.CvCameraViewListener2{
    private static final String tag = "RealTimeActivity";

    CameraBridgeViewBase jcv;

    private String[] names = { "bottle", "book", "router" };
    private Integer [] objIDs = {R.drawable.obj1, R.drawable.obj2, R.drawable.obj3};
    private Integer [] sceneIDs = {R.drawable.single_test1, R.drawable.single_test2, R.drawable.multi_test};

    private double [] scales = { .95, .8, .8 };
    private long startTime, stopTime, elapsedTime;

    //Camera variables
    private Mat mRGB, mGray;
    private Mat imageObj;

    //object detection references
    private Rect touchRect;
    private Point touchPoint;

    private boolean isObjectSelected = false;
    private boolean drawSelectedRect = false;

    FeatureDetector detector;
    DescriptorExtractor extractor;

    MatOfKeyPoint objKeypoints;
    DescriptorMatcher matcher;

    Mat objDescriptor;

    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status){
                case BaseLoaderCallback.SUCCESS: {
                    Log.d(tag, "OpenCV loaded successfully");
                    jcv.enableFpsMeter();
                    jcv.enableView();
                    detector = FeatureDetector.create(FeatureDetector.ORB);
                    extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
                    matcher = DescriptorMatcher.create(4);
                    break;
                }
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_realtime);
        jcv = (CameraBridgeViewBase) findViewById(R.id.java_camera_view);
        jcv.setVisibility(SurfaceView.VISIBLE);
        jcv.setCvCameraViewListener(this);

        if(!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallBack);
        } else {
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(jcv != null)
            jcv.disableView();
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(jcv != null)
            jcv.disableView();
    }

    @Override
    protected void onResume(){
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Log.d(tag, "OpenCV not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallBack);


        }else{
            Log.d(tag, "OpenCV loaded");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRGB = new Mat();
        imageObj = Imgcodecs.imread("res/drawable/obj1.png");
    }

    @Override
    public void onCameraViewStopped() {
        mRGB.release();
        if(mGray!=null){
            mGray.release();
            mGray = null;
        }
    }

    public void init()
    {
        detector.detect(imageObj, objKeypoints);
        extractor.compute(imageObj, objKeypoints, objDescriptor);
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mGray = inputFrame.gray();
        mRGB = inputFrame.rgba();
        Log.d("ONCAMERAFRAME", "inside method");

        MatOfKeyPoint matKeypoints = new MatOfKeyPoint(); //environment keypoints
        objKeypoints = new MatOfKeyPoint(); //object keypoints
        detector.detect(mRGB, matKeypoints);

        Mat envDesc = new Mat(); //detects objects
        extractor.compute(mRGB, matKeypoints, envDesc);

        List<MatOfDMatch> matches = new ArrayList<>();
        List<KeyPoint> keyPointList; //object keypoint list

        objDescriptor = new Mat();

        detector.detect(imageObj, matKeypoints);
        extractor.compute(imageObj, matKeypoints, objDescriptor);
        matcher.knnMatch(objDescriptor, mRGB, matches, 2);

        keyPointList = matKeypoints.toList();

        /*filtering object matches*/
        List<DMatch> filtered = new LinkedList();
        for (int k = 0; k < matches.size(); k++)
        {
            List<DMatch> dm_list = matches.get(k).toList();
            if (dm_list.get(0).distance / dm_list.get(1).distance < scales[1])
            {
                filtered.add(dm_list.get(0));
            }
        }

        //Chooses only best image matches
        if (filtered.size() < 10) {
            return mRGB;
        }

        List<KeyPoint> obj_keypoints_list = matKeypoints.toList();

        LinkedList<Point> objllist = new LinkedList();
        LinkedList<Point> envllist = new LinkedList();

        for (int k = 0; k < filtered.size(); k++)
        {
            objllist.addLast(obj_keypoints_list.get(filtered.get(k).queryIdx).pt);
            envllist.addLast(keyPointList.get(filtered.get(k).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(objllist);

        MatOfPoint2f env = new MatOfPoint2f();
        env.fromList(envllist);

        startTime = System.currentTimeMillis();
        Mat hg = Calib3d.findHomography(obj, env, Calib3d.RANSAC, Calib3d.FM_RANSAC);
        stopTime = System.currentTimeMillis();
        elapsedTime = stopTime - startTime;
        Log.d(tag, "HOMOGRAPHY: " + elapsedTime);

        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2);
        Mat envCorners = new Mat(4, 1, CvType.CV_32FC2);

        objCorners.put(0, 0, new double[]{0, 0});
        objCorners.put(1, 0, new double[]{imageObj.cols(), 0});
        objCorners.put(2, 0, new double[]{imageObj.cols(), imageObj.rows()});
        objCorners.put(3, 0, new double[]{0, imageObj.rows()});

        startTime = System.currentTimeMillis();
        Core.perspectiveTransform(objCorners, envCorners, hg);
        stopTime = System.currentTimeMillis();
        elapsedTime = stopTime - startTime;
        Log.d(tag, "PERSPECTIVETRANSFORM: " + elapsedTime);

        Imgproc.line(mRGB, new Point(envCorners.get(0, 0)), new Point(envCorners.get(1, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(mRGB, new Point(envCorners.get(1, 0)), new Point(envCorners.get(2, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(mRGB, new Point(envCorners.get(2, 0)), new Point(envCorners.get(3, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(mRGB, new Point(envCorners.get(3, 0)), new Point(envCorners.get(0, 0)), new Scalar(0, 255, 0), 4);

        return mRGB;
    }

    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRGB.cols();
        int rows = mRGB.rows();

        int xOffset = (jcv.getWidth() - cols) / 2;
        int yOffset = (jcv.getHeight() - rows) / 2;

        int x = (int) event.getX() - xOffset;
        int y = (int) event.getY() - yOffset;

        touchPoint = new Point(x, y);

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        touchRect = new Rect();

        touchRect.x = (x > 128) ? x - 128 : 0;
        touchRect.y = (y > 128) ? y - 128 : 0;

        touchRect.width = (x + 128 < cols) ? x + 128 - touchRect.x : cols - touchRect.x;
        touchRect.height = (y + 128 < rows) ? y + 128 - touchRect.y : rows - touchRect.y;

        drawSelectedRect = true;
        Mat imageObjectRgba = mRGB.submat(touchRect);
        mGray = new Mat(imageObjectRgba.height(), imageObjectRgba.width(), CvType.CV_8UC1);
        Imgproc.cvtColor(imageObjectRgba, mGray, Imgproc.COLOR_RGB2GRAY);

        isObjectSelected = true;

        return true;
    }

}

