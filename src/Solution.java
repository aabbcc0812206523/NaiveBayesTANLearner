/**
 * Main class which learns and perform classification.
 * 
 * @author Prakhar PAnwaria
 * @date 11/22/2013
 * @hw 3
 */


/*
 * Solution.java
 * 
 * Program accept three command-line arguments as follows: dt-learn
 * <train-set-file> <test-set-file> m
 * 
 * where, train-set-file = Training Set Filename,
 * 		  test-set-file = Test Set Filename,
 * 		  m = threshold value used as a stopping criteria
 */
public class Solution
{

	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		// Program expects three command-line arguments.
		if (args.length != 3 || (!args[2].equals("n") && !args[2].equals("t")) )
		{
			System.err
			.println("Error: Insufficient number of arguments, or invalid 3rd argument."
					+ "\nRequired Arguments: <train_Data_file> <test_data_file> <n|t>");
			System.exit(1);
		}
		
		// Read the file names.
		String trainFilename = args[0];
		String testFilename = args[1];

		// Read examples from the files.
		DataSet trainSet = new DataSet();
		DataSet testSet = new DataSet();
		
		if (!trainSet.ReadInExamplesFromFile(trainFilename)
				|| !testSet.ReadInExamplesFromFile(testFilename))
		{
			System.err.println("Error: Not able to read the datasets.");
			System.exit(1);
		}
		
		// Decide on the basis of "n|t" whether consider Naive Bayes on TAN.
//		trainSet.DescribeDataset();
//		testSet.DescribeDataset();
		
		boolean plotLearningCurve = false;
		if(plotLearningCurve)
		{
			runPlotLearningCurve(trainSet, testSet, args[2]);
		}
		else
		{
			if(args[2].equals("n"))
			{
				NaiveBayes nb = new NaiveBayes(trainSet);
				
				// Prepare model
				nb.prepareBasicModel();
				
				// Print Calculated Model Parameters
				nb.printBasicModelParams();
				
				// Test the accuracy of the model
				double testSetAccuracy = nb.testNaiveBayesModel(testSet);
				
				if(Utility.IS_VERBOSE)
					System.out.println("TestSetAccuracy = " + testSetAccuracy);
			}
			else
			{ 
				TANLearner tl = new TANLearner(trainSet);
				
				// Prepare model
				tl.learnBayesNetStructure();
				
				// Test the accuracy of the model
				double testSetAccuracy = tl.testModel(testSet);
				
				if(Utility.IS_VERBOSE)
					System.out.println("TestSetAccuracy = " + testSetAccuracy);
			}
		}
	}
	
	
	private static void runPlotLearningCurve(DataSet trainSet, DataSet testSet, String learner)
	{
		int [] trainingSetSize = {25, 50, 100};
		int numIterations = 4;
		
		for (int i = 0; i < trainingSetSize.length; i++)
		{
			double avgAccuracy = 0.0;
			for(int j = 0; j < numIterations; j++)
			{
				DataSet inputDataSet = (trainingSetSize[i] == trainSet.size()) ? trainSet : Utility.getRandomDataSet(trainSet, trainingSetSize[i]);
				if(learner.equals("n"))
				{
					NaiveBayes nb = new NaiveBayes(inputDataSet);
					nb.prepareBasicModel();

					// Test Model on Test Set
					avgAccuracy += nb.testNaiveBayesModel(testSet);
					
					if(Utility.IS_VERBOSE)
						System.out.println("Iteration[" + j + "] Avg. accuracy=" + avgAccuracy);
				}
				else
				{
					TANLearner tl = new TANLearner(inputDataSet);
					
					// Prepare model
					tl.learnBayesNetStructure();
				}
			}
			
			avgAccuracy /= numIterations;
			System.out.println("For Training Set Size = " + trainingSetSize[i] + " : Avg Accuracy = " + avgAccuracy);			
		}
	}
	
}
