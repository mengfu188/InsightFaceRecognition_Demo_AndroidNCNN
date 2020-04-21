package com.chenty.testncnn;

/**
 * @ClassName:     MainActivity
 * @Description:   MainActiviry to start all other things
 *
 * @author         chenty
 * @version        V1.0
 * @Date           2019.08.16
 */

import android.content.pm.ActivityInfo;
import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    public static AssetManager manager;

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        manager = getAssets();

        Log.d(TAG, "onViewAttachedToWindow: init ssd start");
        FaceDetect.init(MainActivity.manager);
        Log.d(TAG, "onViewAttachedToWindow: init ssd end");

        if(getRequestedOrientation()!=ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE){
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        Log.i(TAG,"onCreate");

        if (null == savedInstanceState) {
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.container, CameraNcnnFragment.newInstance())
                    .commit();
        }

    }
}
