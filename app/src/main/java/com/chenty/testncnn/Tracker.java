package com.chenty.testncnn;

import android.graphics.Bitmap;

public class Tracker {
    static{
        System.loadLibrary("native-lib");
    }

    public static Bitmap gray(Bitmap bitmap){
        int w = bitmap.getWidth(), h = bitmap.getHeight();
        int[] pix = new int[w * h];
        bitmap.getPixels(pix, 0, w, 0, 0, w, h);
        int [] resultPixes = gray(pix, w, h);
        Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);
        result.setPixels(resultPixes, 0, w, 0, 0, w, h);
        return result;
    }

    public static boolean initTrack(Bitmap bitmap, float[] bbox){
        int w = bitmap.getWidth(), h = bitmap.getHeight();
        int[] pix = new int[w * h];
        bitmap.getPixels(pix, 0, w, 0, 0, w, h);
        boolean result= initTrack(pix, w, h, bbox);
//        Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);
//        result.setPixels(resultPixes, 0, w, 0, 0, w, h);

        return result;
    }

    public static float[] updateTrack(Bitmap bitmap, float[] bbox){
        int w = bitmap.getWidth(), h = bitmap.getHeight();
        int[] pix = new int[w * h];
        bitmap.getPixels(pix, 0, w, 0, 0, w, h);
        float [] result = updateTrack(pix, w, h, bbox);
        return result;
    }

    public static native int[] gray(int[] buf, int w, int h);
    public static native boolean initTrack(int[] buf, int w, int h, float[] bbox);
    public static native float[] updateTrack(int[] buf, int w, int h, float[] bbox);
    public static native void reset();


}
