package com.chenty.testncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class FaceDetectV2 {
    public static float[] detect(Bitmap bitmap){
        float[] faceInfo = detectByBitmap(bitmap);
        if(faceInfo == null){
            return null;
        }
        faceInfo[1] = faceInfo[1] / bitmap.getWidth();
        faceInfo[2] = faceInfo[2] / bitmap.getHeight();
        faceInfo[3] = faceInfo[3] / bitmap.getWidth();
        faceInfo[4] = faceInfo[4] / bitmap.getHeight();

        return faceInfo;
    }

    public static native float[] detectByBitmap(Bitmap data);

//    public static native void init(AssetManager manager);

    /**
     *
     * @param manager
     * @param detect_size 0-5
     * @param horizontal true 为横屏(height < width)使用，false为竖屏height > width使用
     */
    public static native void init(AssetManager manager, int detect_size, boolean horizontal);

    static {
        System.loadLibrary("native-lib");
    }

}
