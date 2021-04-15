package xindaface;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class XDFace {
    static {
        System.loadLibrary("xindaface");
    }

    public native String version();
    public native void load(AssetManager assetManager);
    public native float[] run(Bitmap bitmap);
    public native float[] predict(AssetManager assetManager, Bitmap bitmap);
    public native String pictureRecognition();
}
