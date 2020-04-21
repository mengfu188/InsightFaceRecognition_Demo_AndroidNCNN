package com.chenty.testncnn;


import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

class FaceInfoNorm {
    private float[] pointArray;
    private float left;
    private float top;
    private float right;
    private float bottom;
    private float centerX;
    private float centerY;
    private float width;
    private float height;
    private float[] landmark=new float[10];
    public float[] embedding;

    public static FaceInfoNorm creator(float[] pointArray) {
        if (pointArray == null || pointArray.length == 0 || pointArray.length == 1) return null;
        return new FaceInfoNorm(pointArray);
    }


    private FaceInfoNorm(float[] pointArray) {
        this.pointArray = pointArray;
        this.left = pointArray[1];
        this.top = pointArray[2];
        this.right = pointArray[3];
        this.bottom = pointArray[4];

        this.landmark[0] = this.pointArray[5]; //左眼x
        this.landmark[1] = this.pointArray[6]; //左眼y
        this.landmark[2] = this.pointArray[7]; //有眼x
        this.landmark[3] = this.pointArray[8]; //右眼y
        this.landmark[4] = this.pointArray[9]; //鼻子
        this.landmark[5] = this.pointArray[10];
        this.landmark[6] = this.pointArray[11]; //左嘴角
        this.landmark[7] = this.pointArray[12];
        this.landmark[8] = this.pointArray[13]; //右嘴角
        this.landmark[9] = this.pointArray[14];

        this.centerX = (left + right) / 2;
        this.centerY = (top + bottom) / 2;
        this.width = right - left;
        this.height = bottom - top;
    }

    public float[] getPointArray() {
        return pointArray;
    }

    public float getLeft() {
        return left;
    }

    public float getTop() {
        return top;
    }

    public float getRight() {
        return right;
    }

    public float getBottom() {
        return bottom;
    }

    public float getWidth() {
        return width;
    }

    public float getHeight() {
        return height;
    }

    public float getCenterX() {
        return centerX;
    }

    public float getCenterY() {
        return centerY;
    }

    public float[] getLandmark() { return landmark; }

    @Override
    public String toString() {
        return "FaceInfo{" +
                "pointArray=" + Arrays.toString(pointArray) +
                ", left=" + left +
                ", top=" + top +
                ", right=" + right +
                ", bottom=" + bottom +
                ", centerX=" + centerX +
                ", centerY=" + centerY +
                ", width=" + width +
                ", height=" + height +
                ", landmark=" + Arrays.toString(landmark) +
                '}';
    }
}

class FaceInfo {
    private int[] pointArray;
    private int left;
    private int top;
    private int right;
    private int bottom;
    private int centerX;
    private int centerY;
    private int width;
    private int height;
    private int[] landmark=new int[10];
    public float[] embedding;

    public static FaceInfo creator(int[] pointArray) {
        if (pointArray == null || pointArray.length == 0 || pointArray.length == 1) return null;
        return new FaceInfo(pointArray);
    }

    public static FaceInfo creator(float[] pointArray) {
        if (pointArray == null || pointArray.length == 0 || pointArray.length == 1) return null;
        return new FaceInfo(Util.floatToInt(pointArray));
    }

    private FaceInfo(int[] pointArray) {
        this.pointArray = pointArray;
        this.left = pointArray[1];
        this.top = pointArray[2];
        this.right = pointArray[3];
        this.bottom = pointArray[4];

        this.landmark[0] = this.pointArray[5]; //左眼x
        this.landmark[1] = this.pointArray[6]; //左眼y
        this.landmark[2] = this.pointArray[7]; //有眼x
        this.landmark[3] = this.pointArray[8]; //右眼y
        this.landmark[4] = this.pointArray[9]; //鼻子
        this.landmark[5] = this.pointArray[10];
        this.landmark[6] = this.pointArray[11]; //左嘴角
        this.landmark[7] = this.pointArray[12];
        this.landmark[8] = this.pointArray[13]; //右嘴角
        this.landmark[9] = this.pointArray[14];

        this.centerX = (left + right) / 2;
        this.centerY = (top + bottom) / 2;
        this.width = right - left;
        this.height = bottom - top;
    }

    public int[] getPointArray() {
        return pointArray;
    }

    public int getLeft() {
        return left;
    }

    public int getTop() {
        return top;
    }

    public int getRight() {
        return right;
    }

    public int getBottom() {
        return bottom;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getCenterX() {
        return centerX;
    }

    public int getCenterY() {
        return centerY;
    }

    public int[] getLandmark() { return landmark; }

    @Override
    public String toString() {
        return "FaceInfo{" +
                "pointArray=" + Arrays.toString(pointArray) +
                ", left=" + left +
                ", top=" + top +
                ", right=" + right +
                ", bottom=" + bottom +
                ", centerX=" + centerX +
                ", centerY=" + centerY +
                ", width=" + width +
                ", height=" + height +
                ", landmark=" + Arrays.toString(landmark) +
                '}';
    }
}


class Util {
    private static String TAG="Util";

    public static int[] floatToInt(float[] array){
        int[] ret = new int[array.length];
        for (int i = 0; i < ret.length; i++){
            ret[i] = (int)array[i];
        }
        return ret;
    }

    public static float[] int2Float(int[] array){
        float[] ret = new float[array.length];
        for (int i = 0; i < ret.length; i++){
            ret[i] = (int)array[i];
        }
        return ret;
    }

    public static Bitmap readImageFromAsset(AssetManager asset, String fileName) throws IOException {
        Bitmap bitmap = null;
        InputStream ims = asset.open(fileName);
        bitmap =  BitmapFactory.decodeStream(ims);
        Log.d(TAG, "bitmap config is " + bitmap.getConfig().toString());
        return bitmap;
    }

    public static Bitmap drawFaceInfo(Bitmap bitmap, FaceInfo faceInfo){
        Bitmap drawBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        Canvas canvas = new Canvas(drawBitmap);

        int[] info = faceInfo.getPointArray();

        Paint paint = new Paint();
        int left, top, right, bottom;
        left = (int) info[1];
        top = (int) info[2];
        right = (int) info[3];
        bottom = (int) info[4];
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);//不填充
        paint.setStrokeWidth(5);  //线的宽度
        canvas.drawRect(left, top, right, bottom, paint);
        canvas.drawPoints(new float[]{
                info[5], info[6],
                info[7], info[8],
                info[9], info[10],
                info[11], info[12],
                info[13], info[14]
        }, paint);//画多个点

        return drawBitmap;
    }


}
