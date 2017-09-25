package aymanzeine.objectrecog;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import java.util.ArrayList;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private static final String tag = "MainActivity";

    private ArrayList<String> objNames = new ArrayList(Arrays.asList("book", "sanitizer", "router", "all objects"));
    private ArrayList<String> sceneNames = new ArrayList(Arrays.asList("test1", "test2", "multitest"));

    private int objIndex;
    private int sceneIndex;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void searchImage(View v)
    {
        Intent intent = new Intent(this, ImageDetection.class);
        //intent.putExtra("objIndex", objIndex);
        //intent.putExtra("sceneIndex", sceneIndex);
        startActivity(intent);
    }

    public void searchRealTime(View v)
    {
        Log.d(tag, "SEARCHREALTIME");
        Intent intent = new Intent(this, RealtimeActivity.class);
        //intent.putExtra("obj", objIndex);
        startActivity(intent);
    }

}