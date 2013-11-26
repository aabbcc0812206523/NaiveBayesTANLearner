import java.util.ArrayList;


public class NaiveBayes extends Learner
{
	NaiveBayes(DataSet trainSet)
	{
		super(trainSet);
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
	
	/**
	 * Method to test the accuracy of the model generated.
	 * 
	 * @param testSet	Test data set
	 * @return			Accuracy of the generated model over the given test set.
	 */
	public double testNaiveBayesModel(DataSet testSet)
	{
		int correctPredictionCount = 0;
		
		ArrayList<String> classValues = mTrainSet.getOutputFeature().getValues();
		
		// Run through the examples now.
		for(Example e : testSet)
		{
			Double numerator = (double)(mOutputValueCount[0] + 1);//(mExamplesWithFIRSTLabelCount + 1);
			Double denominator = (double)(mOutputValueCount[1] + 1); //(mExamplesWithSECONDLabelCount + 1);
			
			for(int i = 0; i < mNumSoleFeatures; i++)
			{
				String featureValueInExample = e.get(i);
				
				DiscreteFeature f = (DiscreteFeature) mFeatures.get(i);
				int featureValueIndex =  f.valueIndexMap.get(featureValueInExample);	// Locate the feature value
				ArrayList<Integer[]> featureArray = mBasicModelParams.get(i);
				int numFeatureValues = f.getNumValues();	// Number of values for current feature, used for ensuring Laplace Estimates.
				
				numerator  	*= ((double)(featureArray.get(featureValueIndex)[0] + 1)/(mOutputValueCount[0] + numFeatureValues)); //(mExamplesWithFIRSTLabelCount + numFeatureValues));
				denominator *= ((double)(featureArray.get(featureValueIndex)[1] + 1)/(mOutputValueCount[1] + numFeatureValues));
			}
			
			// Predict
			String predictedLabel = "";
			double posteriorProb = 0.0;
			if(numerator.compareTo(denominator) > 0)	// Predicted Label = FirstLabel
			{
				predictedLabel = classValues.get(0);
				posteriorProb = numerator / (numerator + denominator);
			}
			else
			{
				predictedLabel = classValues.get(1);
				posteriorProb = denominator / (numerator + denominator);
			}

			String actualLabel =  e.get(mNumFeatures - 1);
			if(Utility.IS_VERBOSE)
				System.out.println("e[" + e.getName() + "] :\t" + predictedLabel + "  " + actualLabel);
			else
				System.out.println(predictedLabel + "  " + actualLabel + " " + posteriorProb);
			
			if(predictedLabel.equals(actualLabel))
				correctPredictionCount++;
		}
		
		if(Utility.IS_VERBOSE)
			System.out.println("\nNumber of Correct Predictions = " + correctPredictionCount);
		else
			System.out.println("\n" + correctPredictionCount);
		
		return ((double)correctPredictionCount * 100/testSet.size());
	}
	

}
