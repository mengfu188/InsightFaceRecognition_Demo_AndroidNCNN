package com.chenty.testncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class FaceDetect {

    public static int MAX_SIZD = 320;

    public static Bitmap resize(Bitmap bitmap) {
        // scale
        float long_side = Math.max(bitmap.getWidth(), bitmap.getHeight());
        float scale = MAX_SIZD / long_side;
        int target_height = (int) (bitmap.getHeight() * scale);
        int target_width = (int) (bitmap.getWidth() * scale);
        bitmap = Bitmap.createScaledBitmap(bitmap, target_width, target_height, false);

        return bitmap;
    }

    public static float[] detect(Bitmap bitmap) {


        int width = bitmap.getWidth(), height = bitmap.getHeight();
        float long_side = Math.max(width, height);

        float scale = MAX_SIZD / long_side;
        int target_height = (int) (bitmap.getHeight() * scale);
        int target_width = (int) (bitmap.getWidth() * scale);
        bitmap = Bitmap.createScaledBitmap(bitmap, target_width, target_height, false);

        float[] faceInfo = detectByBitmap(bitmap);
        if(faceInfo == null)
            return null;
        for (int i = 0; i < faceInfo.length / 15; i++) {
            for (int j = 1; j <= 14; j++) {
                faceInfo[i * 15 + j] = faceInfo[i * 15 + j] / scale;
            }
        }

        for (int i = 0; i < faceInfo.length / 15; i++) {
            // bbox
            faceInfo[i * 15 + 1] = faceInfo[i * 15 + 1] / width;
            faceInfo[i * 15 + 2] = faceInfo[i * 15 + 2] / height;
            faceInfo[i * 15 + 3] = faceInfo[i * 15 + 3] / width;
            faceInfo[i * 15 + 4] = faceInfo[i * 15 + 4] / height;
            // landmark
            for (int j = 5; j <= 14; j+=2) {
                faceInfo[i * 15 + j] = faceInfo[i * 15 + j] / width;
                faceInfo[i * 15 + j + 1] = faceInfo[i * 15 + j + 1] / height;
            }
        }

        return faceInfo;
    }

    public static native float[] detectByBitmap(Bitmap data);

    public static native void setThreshold(float threshold);

    public static native void init(AssetManager manager);

    static {
        System.loadLibrary("native-lib");
    }
}
