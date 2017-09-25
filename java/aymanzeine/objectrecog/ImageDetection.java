package aymanzeine.objectrecog;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static aymanzeine.objectrecog.R.id.imageView;

public class ImageDetection extends AppCompatActivity {

    private static final String tag = "ImageDetectionActivity";

    private Integer [] objIDs = {R.drawable.obj1, R.drawable.obj2, R.drawable.obj3};
    private Integer [] sceneIDs = {R.drawable.single_test1, R.drawable.single_test2, R.drawable.multi_test};

    private int objIndex;
    private int sceneIndex;
    private Integer objId;
    private Integer sceneId;
    private long startTime, stopTime, elapsedTime;


    private double [] scales = {.95, .8, .8};

    private ImageView imgView;
    private Mat objMat;
    private Mat sceneMat;
    private DescriptorMatcher matcher;
    private Bitmap b;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this){
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
                    matcher = DescriptorMatcher.create(4);
                    if (objIndex == 3){
                        Log.d(tag, "Multiple Objects Detected");
                        findMultipleObjectsObject();
                    } else {
                        Log.d(tag, "Single Object Detected " + objIndex);
                        findObject();
                    }
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detection);
        Log.d(tag, "IMAGEDETECTION");
        Button imageA = (Button) findViewById(R.id.Abutton);
        Button imageB = (Button) findViewById(R.id.BButton);
        Button multi = (Button) findViewById(R.id.multiButton);

        imageA.setOnClickListener(new View.OnClickListener(){
            public void onClick(View v) {
                objIndex = 0;
                objId = objIDs[objIndex];
                sceneIndex = 0;
                sceneId = sceneIDs[sceneIndex];
                Log.d(tag, "IMAGE A " + objId);
                findObject();
            }
        });

        imageB.setOnClickListener(new View.OnClickListener(){
            public void onClick(View v) {
                objIndex = 0;
                objId = objIDs[objIndex];
                sceneIndex = 1;
                sceneId = sceneIDs[sceneIndex];
                Log.d(tag, "IMAGE B " + objIndex);
                findObject();
            }
        });

        multi.setOnClickListener(new View.OnClickListener(){
            public void onClick(View v) {
                objIndex = 3;
                sceneIndex = 2;
                sceneId = sceneIDs[sceneIndex];
                Log.d(tag, "IMAGE C " + objId);
                findMultipleObjectsObject();
            }
        });

        if(objIndex < 3){
            objId = objIDs[objIndex];
        }

        sceneId = sceneIDs[sceneIndex];
        Log.d(tag, "SCENE INDEX " + sceneIndex);

        if(!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void findObject(){
        try {
            sceneMat = Utils.loadResource(this, sceneId, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            Log.d(tag, "SceneId = " + sceneIndex);
        }
        catch(IOException e) {
            Log.e(tag, "Error find objects");
        }

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        MatOfKeyPoint sKeypoints = new MatOfKeyPoint();
        detector.detect(sceneMat, sKeypoints);

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        Mat sDescriptor = new Mat();
        extractor.compute(sceneMat, sKeypoints, sDescriptor);

        try{
            objMat = Utils.loadResource(this, objIDs[objIndex], Imgcodecs.CV_LOAD_IMAGE_COLOR);
        }
        catch (IOException e){
            Log.e(tag, "Error find objects post extract");
        }

        /* detect keypoints in object image */
        MatOfKeyPoint objKeypoints = new MatOfKeyPoint();

        List<MatOfDMatch> matches = new ArrayList();
        List<KeyPoint> sKeyList;

        Mat objDescriptor = new Mat();
        detector.detect(objMat, objKeypoints);
        extractor.compute(objMat, objKeypoints, objDescriptor);
        matcher.knnMatch(objDescriptor, sDescriptor, matches, 2);
        sKeyList = sKeypoints.toList();

        /* filter out matches based on hamming distance */
        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();

        for (int k = 0; k < matches.size(); k++){
            List<DMatch> dm_list = matches.get(k).toList();
            if (dm_list.get(0).distance / dm_list.get(1).distance < scales[objIndex]){
                good_matches.addLast(dm_list.get(0));
            }
        }

        if (good_matches.size() < 10){
            Toast.makeText(this, "Unable to detect object", Toast.LENGTH_LONG).show();
            return;
        }

        List<KeyPoint> oKeyList = objKeypoints.toList();

        LinkedList<Point> oKeyLList = new LinkedList<Point>();
        LinkedList<Point> sKeyLList = new LinkedList<Point>();

        for (int k = 0; k < good_matches.size(); k++){
            oKeyLList.addLast(oKeyList.get(good_matches.get(k).queryIdx).pt);
            sKeyLList.addLast(sKeyList.get(good_matches.get(k).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(oKeyLList);

        MatOfPoint2f scene = new MatOfPoint2f();
        scene.fromList(sKeyLList);

        startTime = System.currentTimeMillis();
        Mat hg = Calib3d.findHomography(obj, scene, Calib3d.RANSAC, Calib3d.FM_RANSAC);
        stopTime = System.currentTimeMillis();
        elapsedTime = stopTime - startTime;
        Log.d(tag, "HOMOGRAPHY: " + elapsedTime);

        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2);
        Mat envCorners = new Mat(4, 1, CvType.CV_32FC2);

        objCorners.put(0, 0, new double[]{0, 0});
        objCorners.put(1, 0, new double[]{objMat.cols(), 0});
        objCorners.put(2, 0, new double[]{objMat.cols(), objMat.rows()});
        objCorners.put(3, 0, new double[]{0, objMat.rows()});

        startTime = System.currentTimeMillis();
        Core.perspectiveTransform(objCorners, envCorners, hg);
        stopTime = System.currentTimeMillis();
        elapsedTime = stopTime - startTime;
        Log.d(tag, "PERSPECTIVETRANSFORM: " + elapsedTime);

        Imgproc.line(sceneMat, new Point(envCorners.get(0, 0)), new Point(envCorners.get(1, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(sceneMat, new Point(envCorners.get(1, 0)), new Point(envCorners.get(2, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(sceneMat, new Point(envCorners.get(2, 0)), new Point(envCorners.get(3, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(sceneMat, new Point(envCorners.get(3, 0)), new Point(envCorners.get(0, 0)), new Scalar(0, 255, 0), 4);

        imgView = (ImageView) findViewById(R.id.imageView);

        try{
            Imgproc.cvtColor(sceneMat, sceneMat, Imgproc.COLOR_BGR2RGB);
            Bitmap bmp = Bitmap.createBitmap(sceneMat.cols(), sceneMat.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(sceneMat, bmp);
            imgView.setImageBitmap(bmp);
        }
        catch (CvException e){
            Log.e("MAT_BITMAP EXCEPTION", e.getMessage());
        }
    }

    public void findMultipleObjectsObject(){
        try {
            sceneMat = new Mat();
            b = BitmapFactory.decodeResource(getResources(), sceneId);
            Utils.bitmapToMat(b, sceneMat);
        }
        catch(Exception e) {
            Log.e(tag, "Error multi");
        }

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        MatOfKeyPoint sKeypoints = new MatOfKeyPoint();

        startTime = System.currentTimeMillis();
        detector.detect(sceneMat, sKeypoints);
        stopTime = System.currentTimeMillis();
        elapsedTime += (stopTime - startTime);
        Log.d(tag, "1st DETECTOR: " + elapsedTime);

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        Mat sDescriptor = new Mat();

        startTime = System.currentTimeMillis();
        extractor.compute(sceneMat, sKeypoints, sDescriptor);
        stopTime = System.currentTimeMillis();
        elapsedTime += (stopTime - startTime);
        Log.d(tag, "1st EXTRACTOR: " + elapsedTime);

        //Loop through each object in order to find it in scene
        for (int i = 0; i < 3; i++) {
            /* read obj and scene images*/
            try {
                objMat = new Mat();
                b = BitmapFactory.decodeResource(getResources(), objIDs[i]);
                Utils.bitmapToMat(b, objMat);
            } catch (Exception e) {
                Log.e(tag, "Error find multi objects post extract");
            }

        /* detect keypoints in object image */
            MatOfKeyPoint objKeypoints = new MatOfKeyPoint();

            List<MatOfDMatch> matches = new ArrayList();
            List<KeyPoint> sKeyList;

            Mat objDescriptor = new Mat();

            startTime = System.currentTimeMillis();
            detector.detect(objMat, objKeypoints);
            stopTime = System.currentTimeMillis();
            elapsedTime += (stopTime - startTime);
            Log.d(tag, "DETECTOR: " + elapsedTime);

            startTime = System.currentTimeMillis();
            extractor.compute(objMat, objKeypoints, objDescriptor);
            stopTime = System.currentTimeMillis();
            elapsedTime += (stopTime - startTime);
            Log.d(tag, "EXTRACTOR: " + elapsedTime);

            startTime = System.currentTimeMillis();
            matcher.knnMatch(objDescriptor, sDescriptor, matches, 2);
            stopTime = System.currentTimeMillis();
            elapsedTime += (stopTime - startTime);
            Log.d(tag, "MATCHER: " + elapsedTime);

            sKeyList = sKeypoints.toList();

        /* Filter out matches based on hamming distance to find best image of object in scene */
            LinkedList<DMatch> good_matches = new LinkedList<DMatch>();

            for (int k = 0; k < matches.size(); k++) {
                List<DMatch> dm_list = matches.get(k).toList();
                if (dm_list.get(0).distance / dm_list.get(1).distance < scales[1]) {
                    good_matches.addLast(dm_list.get(0));
                }
            }

            if (good_matches.size() < 10) {
                Toast.makeText(this, "Unable to detect object", Toast.LENGTH_LONG).show();
                return;
            }

            List<KeyPoint> oKeyList = objKeypoints.toList();

            LinkedList<Point> oKeyLList = new LinkedList<Point>();
            LinkedList<Point> sKeyLList = new LinkedList<Point>();

            for (int k = 0; k < good_matches.size(); k++) {
                oKeyLList.addLast(oKeyList.get(good_matches.get(k).queryIdx).pt);
                sKeyLList.addLast(sKeyList.get(good_matches.get(k).trainIdx).pt);
            }

            MatOfPoint2f obj = new MatOfPoint2f();
            obj.fromList(oKeyLList);

            MatOfPoint2f scene = new MatOfPoint2f();
            scene.fromList(sKeyLList);

            startTime = System.currentTimeMillis();
            Mat hg = Calib3d.findHomography(obj, scene, Calib3d.RANSAC, Calib3d.FM_RANSAC);
            stopTime = System.currentTimeMillis();
            elapsedTime = stopTime - startTime;
            Log.d(tag, "HOMOGRAPHY: " + elapsedTime);

            Mat objCorners = new Mat(4, 1, CvType.CV_32FC2);
            Mat envCorners = new Mat(4, 1, CvType.CV_32FC2);

            objCorners.put(0, 0, new double[]{0, 0});
            objCorners.put(1, 0, new double[]{objMat.cols(), 0});
            objCorners.put(2, 0, new double[]{objMat.cols(), objMat.rows()});
            objCorners.put(3, 0, new double[]{0, objMat.rows()});

            startTime = System.currentTimeMillis();
            Core.perspectiveTransform(objCorners, envCorners, hg);
            stopTime = System.currentTimeMillis();
            elapsedTime = stopTime - startTime;
            Log.d(tag, "PERSPECTIVETRANSFORM: " + elapsedTime);

            Imgproc.line(sceneMat, new Point(envCorners.get(0, 0)), new Point(envCorners.get(1, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(sceneMat, new Point(envCorners.get(1, 0)), new Point(envCorners.get(2, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(sceneMat, new Point(envCorners.get(2, 0)), new Point(envCorners.get(3, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(sceneMat, new Point(envCorners.get(3, 0)), new Point(envCorners.get(0, 0)), new Scalar(0, 255, 0), 4);
        }

        imgView = (ImageView) findViewById(R.id.imageView);

        try{
            Bitmap bmp = Bitmap.createBitmap(sceneMat.cols(), sceneMat.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(sceneMat, bmp);
            imgView.setImageBitmap(bmp);
        }
        catch (CvException e){
            Log.d("MAT_TO_BITMAP_EXCEPTION", e.getMessage());
        }
    }
}
