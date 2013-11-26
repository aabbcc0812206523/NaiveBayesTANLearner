import java.util.ArrayList;


public class Learner
{
	public DataSet mTrainSet = null;

	Learner(DataSet trainSet)
	{
		mTrainSet = trainSet;
	}
	
	/**
	 * Method to initializew the model parameters.
	 */
	void initBasicModelParams()
	{
		mNumFeatures = mTrainSet.getNumberOfFeatures();
		mNumSoleFeatures = mNumFeatures - 1;
		mFeatures = mTrainSet.getFeatures();
		mOutputValueCount = new Integer[2]; mOutputValueCount[0] = 0; mOutputValueCount[1] = 0; 
		
		for(int i = 0; i < mNumSoleFeatures; i++) // 'numFeatures - 1', because ignoring Class Label
		{
			DiscreteFeature f = (DiscreteFeature) mFeatures.get(i);
			
			// ASSUMPTION: All features are discrete valued.
			int numValues = f.getNumValues();
			ArrayList<Integer[]> featureArrayList = new ArrayList<Integer[]>();
			
			for(int j = 0; j < numValues; j++)
			{
				Integer[] values = new Integer[2];  // '2' because output label is assumed binary
				values[0] = 0; values[1] = 0;
				featureArrayList.add(values);
			}
			
			mBasicModelParams.add(featureArrayList);
		}
	}
	
	/**
	 * Main method for Naive Bayes Solution
	 */
	public void prepareBasicModel()
	{
		// Focus only on Training Set
		
		// Go through all the features of the data set (except class)
		if(Utility.IS_VERBOSE)
			mTrainSet.DescribeDataset();
		
		// Initialize ModelParams
		initBasicModelParams();
		
		String classFirstValue = mTrainSet.getOutputFeature().getValues().get(0);
		
		// Run through the examples now.
		for(Example e : mTrainSet)
		{
			// ASSUMPTION: Class label is binary-valued.
			int classLabelIndex = e.get(mNumFeatures - 1).equals(classFirstValue) ? 0 : 1;
			
			if(classLabelIndex == 0) 
				mOutputValueCount[0]++;
//				mExamplesWithFIRSTLabelCount++;
			else
				mOutputValueCount[1]++;
				
			
			// Go through all the feature values of this training
			// set instance and update conditional probability tables.
			for(int i = 0; i < mNumSoleFeatures; i++)
			{
				String featureValueInExample = e.get(i);
				
				DiscreteFeature f = (DiscreteFeature) mFeatures.get(i);
				int featureValueIndex =  f.valueIndexMap.get(featureValueInExample);	// Locate the feature value

				mBasicModelParams.get(i).get(featureValueIndex)[classLabelIndex]++;	
			}
		}
	}
	
	/**
	 * Method to print the model parameters.
	 */
	void printBasicModelParams()
	{
		int i = 0;
		
		if(Utility.IS_VERBOSE)
		{
			for(ArrayList<Integer[]> featureArray : mBasicModelParams)
			{
				System.out.println("Feature[" + i++ + "]\n-----------------");
				
				int j = 0;
				for(Integer[] values : featureArray)
					System.out.println("FeatureValue[" + j++ + "] = " + values[0] + " | " + values[1]);
				
				System.out.println();
			}
		}
		
		i = 0;
		for(; i < mNumSoleFeatures; i++)
			System.out.println(mFeatures.get(i).getName() + " class");
		System.out.println();
	}
	
	/* Member Variables */
	
	// P(Xi|Y), for all Features (Xi) 
	public ArrayList<ArrayList<Integer[]>> mBasicModelParams = new ArrayList<ArrayList<Integer[]>>();
	
	// P(Y)
	public Integer[] mOutputValueCount = null;
//	public int mExamplesWithFIRSTLabelCount = 0;
	
	// Caching Training Set information
	public int mNumFeatures = 0;
	public int mNumSoleFeatures = 0;
	public ArrayList<Feature> mFeatures = null;
	
}
