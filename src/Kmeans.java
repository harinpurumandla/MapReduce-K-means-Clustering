import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.hdfs.DistributedFileSystem;

public class Kmeans {
	private static double [][] _centroids;
	private static int itr=0;
	private static String CENTROID="";
	
//DFS location in my system

	private static String dfsLocation="hdfs://localhost:9000/";
	
//Calculating the Euclidian distance. Reference :  http://cecs.wright.edu/~keke.chen/cloud/2016/labs/mapreduce/KMeans.java 	

	 private static double dist(double [] v1, double [] v2){
		    double sum=0;
		    for (int i=0; i<v1.length; i++){
		      double d = v1[i]-v2[i];
		      sum += d*d;
		    }
		    return Math.sqrt(sum);
		  }
// Convergenction condition: When centroids in previous iteration and current interation are same then converge the program and current centroids are final centroids		  
		  
	private static boolean converge(double [][] c1, double [][] c2){
	    // c1 and c2 are two sets of centroids 
	   if(Arrays.deepEquals(c1, c2))
		   return true;
	   else
		   return false;
	    
	  }
	  
// Mapper function

	public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {

//Setup for reading the centroids file

	    @Override
	    protected void setup(Context context) throws IOException, InterruptedException {
	    	Configuration conf = context.getConfiguration();
	    	_centroids=readCentroids(conf,CENTROID);
	    }
		
// After reading the centroids from centroids file, reading the input file and calculating the minimum distance and grouping the clusters using cluster_id
   
	    @Override
	    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
	    	double [][] centroids=_centroids;
	        String[] line = value.toString().split(",");
	        double x = Double.parseDouble(line[0]);
	        double y = Double.parseDouble(line[1]);
	        int cluster_id = 0;
	        double minDistance = -1;
	        for (int i = 0; i < centroids[0].length; i++) {
	            double distance = dist(centroids[i],new double[]{x,y});
	            if (distance < minDistance || minDistance == -1) {
	                cluster_id = i;
	                minDistance = distance;
	            }
	        }

	        context.write(new IntWritable(cluster_id), value);
	    }
	}
	
// In Reducer class, we will get all the records with same cluster_id aggregated to reducer and reducer class will calculate the new centroids  	

	public static class KMeansReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

	    protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
	    	double sum_x=0,sum_y=0;
	    	int number_records=0;
	    	for(Text value: values)
	    	{
	    		
	    	String[] line = value.toString().split(",");
	        sum_x += Double.parseDouble(line[0]);
	        sum_y += Double.parseDouble(line[1]);
	        number_records++;
	    	}
	    	sum_x /=(double)number_records;
	    	sum_y /=(double)number_records;
	    	String centroid= sum_x+","+sum_y;
	    	context.write(key, new Text(centroid));
	    	
		}
	}

	public static void main(String[] args) throws Exception {
		if(args.length>4 || args.length<3) //asking for hdfs port if the port is not 9000
		{
			System.out.println("usage: Kmean <input> <output> <centroid-file> <port(optional)>");
			System.exit(0);
		}
		boolean isconverged=false;
		double[][] centroid;
		
		if(args.length==4){
				dfsLocation="hdfs://localhost:"+args[3]+"/";
			
		}
		CENTROID=dfsLocation+args[2];
		copyTemp(new Configuration(),CENTROID);
		do{
			itr++;
//Getting configuration of Hadoop.			
		Configuration conf = new Configuration();	
		Job job = Job.getInstance(conf); //Creating the job
		job.setJobName("KMeans");
        job.setJarByClass(Kmeans.class); //Assining the jar class for main execution.
        job.setMapperClass(KMeansMapper.class); //Assining the Mapper class
        job.setReducerClass(KMeansReducer.class); //Assining the Reducer class
        job.setMapOutputKeyClass(IntWritable.class); //Defining the output key data type of mapper 
        job.setMapOutputValueClass(Text.class); //Defining the output value data type of mapper 
		FileInputFormat.setInputPaths(job, new Path(dfsLocation+args[0])); //Input from DFS location 
		FileOutputFormat.setOutputPath(job, new Path(dfsLocation+args[1])); //Output to DFS location
		
//After one iteration of the job, we will check for the convergence condition, if it satisfies then terminate the program with current centroids as final centroids
// and if not satisfies then copy the output of this job to the centriods file and delete the current output and restart the job with new centriods.
 		
		job.waitForCompletion(true);
		centroid=readCentroids(conf, dfsLocation+args[1]+"/"+"part-r-00000"); // Reading the output of the job to centroid array
		if(converge(_centroids, centroid))  //Checking the converge condition
		{
			removeTemp(conf, CENTROID);	//Removing the temporary file.
			isconverged=true; 
		}
		else
		{
			copyCentroid(conf,dfsLocation+args[1]);
		deleteFile(conf,dfsLocation+args[1]);
			isconverged=false;
		}
		}while(!isconverged);
		
	}
	
// Deleting the output file
	
	public static boolean deleteFile(Configuration conf,String output) throws IOException
	{
		FileSystem fs = new DistributedFileSystem();
        try {
			fs.initialize(new URI(dfsLocation), conf);
			if(fs.delete(new Path(output),true))
			{
				fs.close();
				return true;
			}
			else{
				fs.close();
				return false;
			}
			
		} catch (URISyntaxException e1) {
			// TODO Auto-generated catch block
			fs.close();
			return false;
		}
		
	}
	
// Copy the output of the job to new centriods file
	
	public static boolean copyCentroid(Configuration conf,String output) throws IOException
	{
		FileSystem fs = new DistributedFileSystem();
        try {
			fs.initialize(new URI(dfsLocation), conf);
			FileUtil.copy(fs, new Path(output+"/part-r-00000"), fs, new Path(CENTROID), true, new Configuration());
			fs.close();
			return true;
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			fs.close();
			return false;
		}
	}
	
// Copying the temp file to centroid file and removing the temporory centroid file.
	
	public static boolean removeTemp(Configuration conf,String output) throws IOException
	{
		FileSystem fs = new DistributedFileSystem();
        try {
			fs.initialize(new URI(dfsLocation), conf);
			FileUtil.copy(fs, new Path(CENTROID+"-temp"), fs, new Path(CENTROID), true, new Configuration());
			fs.close();
			return true;
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			fs.close();
			return false;
		}
	}
//Coping the original centroid file to main file.
	
	public static boolean copyTemp(Configuration conf,String output) throws IOException
	{
		FileSystem fs = new DistributedFileSystem();
        try {
			fs.initialize(new URI(dfsLocation), conf);
			FileUtil.copy(fs, new Path(CENTROID), fs, new Path(CENTROID+"-temp"), false, new Configuration());
			fs.close();
			return true;
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			fs.close();
			return false;
		}
	}
	
//Acessing the centriods file from HDFS location and reading it to an array _centroids 	
	public static double[][] readCentroids(Configuration conf,String loc)throws IOException, InterruptedException
	{
		
        // get the HDFS filename from the conf object 
		
    	String value;
        Path path = new Path(loc);  
        FileSystem fs = new DistributedFileSystem();
        int i=0;
        try {
			fs.initialize(new URI(dfsLocation), conf);
			
		} catch (URISyntaxException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
        try{
            BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));
            ArrayList<double[]> arr = new ArrayList<double[]>();
            
            while ((value = br.readLine()) != null) {

            	String[] line = value.split("\t")[1].split(",");
            	double[] temp=new double[2];
            	temp[0] = Double.parseDouble(line[0]);
            	temp[1] = Double.parseDouble(line[1]);
            	arr.add(temp);
    	        i++;
            }
            double[][] centroid = new double[arr.size()][];
            for(int j=0;j<arr.size();j++)
            {
            	centroid[j]=arr.get(j);
            }
           
            fs.close();
            return centroid;
            //do something with the file content
         }catch(Exception e){
        	 return null;
         }
	}
}