
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


class DetectFaceDemo {
	
	  public void run() {
	    System.out.println("\nRunning NMI");
	    float resizeImg=0.015625f;
	    int num=256;
	    double h1,h2;
	    System.out.println("resizeImg="+resizeImg);
	    //Pic1
	    Mat image1 = Imgcodecs.imread(getClass().getResource("/eS001.0-31_Frame1.jpg").getPath());
	    Mat yuvimg1 = new Mat(image1.height(),image1.width(), CvType.CV_8U);
	    Mat smallPic1 = new Mat(image1.height(),image1.width(), CvType.CV_8U);
	    
	    Size dsize1=image1.size();
	    dsize1.height=dsize1.height*resizeImg;
	    dsize1.width=dsize1.width*resizeImg;
	    Imgproc.resize(image1, smallPic1, dsize1);
	    Imgproc.cvtColor(smallPic1, yuvimg1, Imgproc.COLOR_RGB2YUV);
	    List<Mat> Yimg1 = new ArrayList<Mat>(3);
	    Core.split(yuvimg1, Yimg1);

	    Mat channel1 = Yimg1.get(0);
	    h1=entropy(channel1,num);
	    System.out.println("h1="+h1);
	    
	    //Pic2
	    Mat image2 = Imgcodecs.imread(getClass().getResource("/eS001.0-31_Frame2.jpg").getPath());
	    Mat yuvimg2 = new Mat(image2.height(),image2.width(), CvType.CV_8U);
	    Mat smallPic2 = new Mat(image2.height(),image2.width(), CvType.CV_8U);
	    
	    Size dsize2=image1.size();
	    dsize2.height=dsize2.height*resizeImg;
	    dsize2.width=dsize2.width*resizeImg;
	    Imgproc.resize(image2, smallPic2, dsize2);
	    Imgproc.cvtColor(smallPic2, yuvimg2, Imgproc.COLOR_RGB2YUV);
	    List<Mat> Yimg2 = new ArrayList<Mat>(3);
	    Core.split(yuvimg2, Yimg2);

	    Mat channel2 = Yimg2.get(0);
	    h2=entropy(channel2,num);
	    System.out.println("h2="+h2);
	    
	    int[][] h=jointH(pix(channel1,num),pix(channel2,num),num);
	    double sum=0.0;
	    int[] n=new int[h.length*h.length];
	    int u=0;
	    for(int i=0;i<256;i++){
			  for(int j=0;j<256;j++){
				  sum+=h[i][j];
				  n[u++]=h[i][j];
				  }
			  }
	    double hAB=entropy1(n,sum,num);
	    double enMI=h1+h2-hAB;
	    double enNMI=(h1+h2)/hAB;
	    System.out.println("H(A,B)= "+hAB);
	    System.out.println("MI(A,B)= "+enMI);
	    System.out.println("NMI(A,B)= "+enNMI);
	    // Save the visualized detection.
//	    String filename = "yuvimg.png";
//	    System.out.println(String.format("Writing %s", filename));
//	    Imgcodecs.imwrite(filename, yuvimg);
	  }
	  public int[] section(int n){
		  int diff=0;
		  int[] num=new int[n];
		  int t=0;
		  diff=256/n;
		  int temp =diff -1;
		  for(int x=0;x<n;x++){
			  int b=t;
			  t=t+temp;
			  num[x]=t;
			  t++;  
		  }
		  return num;
	  }
	  public int[][] jointH(Mat img1,Mat img2,int n){
		  int[][] count = new int[256][256];
		  int X=img1.width();
		  int Y=img1.height();
		  int a=0,b=0;
		  
		  for(int i=0;i<Y;i++){
			  for(int j=0;j<X;j++){
				  a=(int) img1.get(i, j)[0];
				  b=(int) img2.get(i, j)[0];
				  count[a][b]++;
			  }
		  }
		return count;
		  
	  }
	  public Mat pix(Mat img1,int n) {
		  int diff=0;
		  int X=img1.width();
		  int Y=img1.height();
		  Mat img2 = new Mat(img1.height(),img1.width(), CvType.CV_8U);
		  double a=0;
		  for(int i=0;i<Y;i++){
			  for(int j=0;j<X;j++){
				  a=img1.get(i,j)[0];
				  double data;
				  double t=0.0;
				  diff =256/n;
				  double temp = diff-1;
				  double b;
				  for(int q=0;q<n;q++){
					  b=t;
					  t=t+temp;
					  if(a<= t && a>=b){
						data=t;
						img2.put(i, j, data);
					  }
					  t=t+1;
				  }
			  }
		  }
		return img2;
		  
	  }
	  
	  public double entropy(Mat img1,int n) {
		  int diff=0;
		  int X=img1.width();
		  int Y=img1.height();
		  Mat img2 = new Mat(img1.height(),img1.width(), CvType.CV_8U);
		  double a=0;
		  int[] count = new int[256];
		  
		  for(int i=0;i<Y;i++){
			  for(int j=0;j<X;j++){
				  a=img1.get(i,j)[0];
				  double data;
				  double t=0.0;
				  diff =256/n;
				  double temp = diff-1;
				  double b;
				  for(int q=0;q<n;q++){
					  b=t;
					  t=t+temp;
					  if(a<= t && a>=b){
						   data=t;
						img2.put(i, j, data);
						count[(int) t]++;
						
					  }
					  t=t+1;
					  
				  }
			  }
		  }
		  return entropy1(count,sumCount(count),n);
	  }
	  private double sumCount(int[] count) {
			double sum=0.0;
			for(int i=0;i<count.length;i++)
				sum+=count[i];
			return sum;
		}
	private double entropy1(int[] count, double space, int n) {
		double h=0.0;
		double sum=0.0;
		for(int i=0;i<count.length;i++){
			if(count[i]/space!=0)
			{
				double p=count[i]/space;
				h=-p*Math.log10(p);
				sum+=h;
			}
			
		}
		return sum;
	}
	  
	}
public class NMIproject {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("Hello, OpenCV");
	    // Load the native library.
	    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	    new DetectFaceDemo().run();
	}

}

