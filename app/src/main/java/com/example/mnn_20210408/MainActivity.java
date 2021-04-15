package com.example.mnn_20210408;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

import xindaface.XDFace;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private ArrayList<String> classNames;
    XDFace xdFace;
    ImageView iv;
    TextView tv;
    Button btn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        iv = findViewById(R.id.iv);
        tv = findViewById(R.id.tv);
        btn = findViewById(R.id.btn);
        classNames = ReadListFromFile(getAssets(), "synset_words.txt");
        xdFace = new XDFace();

        test_lr();
    }

    public class Cats {
        public float score;
        String name;

        public Cats(float score, String name) {
            this.score = score;
            this.name = name;
        }
    }

    public void test() {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.raw.testcat, options);
        AssetManager assetManager = getAssets();
        float[] results = xdFace.predict(assetManager, bitmap);
        PriorityQueue<Cats> pq = new PriorityQueue<>(new Comparator<Cats>() {
            @Override
            public int compare(Cats o1, Cats o2) {
                if (o2.score<o1.score) return -1;
                if (o2.score==o1.score) return 0;
                return 1;
            }
        });
        for (int i=0; i<classNames.size(); i++) {
            Cats cats = new Cats(results[i], classNames.get(i));
            pq.add(cats);
        }
        for (int i=0; i<5; i++) {
            Cats cats = pq.poll();
            Log.i(TAG, cats.name + " " + cats.score);
        }
    }

    public void test_lr() {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        final Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.raw.testcat, options);
        iv.setImageBitmap(bitmap);

        AssetManager assetManager = getAssets();
        xdFace.load(assetManager);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                long start = System.currentTimeMillis();
                for (int j=1; j<10; j++) {
                    float[] results = xdFace.run(bitmap);
                    PriorityQueue<Cats> pq = new PriorityQueue<>(new Comparator<Cats>() {
                        @Override
                        public int compare(Cats o1, Cats o2) {
                            if (o2.score<o1.score) return -1;
                            if (o2.score==o1.score) return 0;
                            return 1;
                        }
                    });
                    for (int i=0; i<classNames.size(); i++) {
                        Cats cats = new Cats(results[i], classNames.get(i));
                        pq.add(cats);
                    }
                    for (int i=0; i<5; i++) {
                        Cats cats = pq.poll();
                        Log.i(TAG, cats.name + " " + cats.score);
                    }
                }
                long end = System.currentTimeMillis();
                Log.i(TAG, "time cost = " + (end-start)/10);
                tv.setText("time cost = " + (end-start)/10);
            }
        });

    }

    public ArrayList<String> ReadListFromFile(AssetManager assetManager, String filePath) {
        ArrayList<String> list = new ArrayList<String>();
        BufferedReader reader = null;
        InputStream istr = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(assetManager.open(filePath)));
            String line;
            while ((line = reader.readLine()) != null) {
                list.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }
}
