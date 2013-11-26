/**
 * Main class which learns and perform classification using either Naive Bayes or TAN Learner.
 * 
 * @author Prakhar PAnwaria
 * @date 11/22/2013
 * @hw 3
 */

/*
 * Solution.java
 * 
 * Program accept three command-line arguments as follows: bayes
 * <train-set-file> <test-set-file> <n|t>
 * 
 * where, train-set-file = Training Set Filename,
 * 		  test-set-file = Testing Set Filename,
 * 		  n or t = 'n' for Naive Bayes, 't' for TAN Learner
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
				double testSetAccuracy = nb.testNaiveBayesModel(testSet, true);
				
				if(Utility.IS_VERBOSE)
					System.out.println("TestSetAccuracy = " + testSetAccuracy);
			}
			else
			{ 
				TANLearner tl = new TANLearner(trainSet);
				
				// Learn Bayes Net Structure
				tl.learnBayesNetStructure();
				
				// Print Bayes Net Structure
				tl.printBayesNetStructure();
				
				// Test the accuracy of the model
				double testSetAccuracy = tl.testBayesNetStructure(testSet, true);
				
				if(Utility.IS_VERBOSE)
					System.out.println("TestSetAccuracy = " + testSetAccuracy);
			}
		}
	}
	
	/**
	 * Method to get avg accuracy of the learned models on randomly generated training sets.
	 * @param trainSet	Original Training Set
	 * @param testSet	Given Test Set
	 * @param learner	Learning Method ('n' for Naive Bayes, 't' for TAN Learner)
	 */
	private static void runPlotLearningCurve(DataSet trainSet, DataSet testSet, String learner)
	{
		int [] trainingSetSize = {25, 50, 100};
		int numIterations = 4;
		
		System.out.println(learner.equals("n")? "Method: Naive Bayes\n" : "Method: TAN\n");
		
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

					// Test the accuracy of the model on Test Set
					avgAccuracy += nb.testNaiveBayesModel(testSet, false);
				}
				else
				{
					TANLearner tl = new TANLearner(inputDataSet);
					tl.learnBayesNetStructure();
					
					// Test the accuracy of the model on Test Set
					avgAccuracy += tl.testBayesNetStructure(testSet, false);
				}
				
				if(Utility.IS_VERBOSE)
					System.out.println("Iteration[" + j + "] Avg. accuracy=" + avgAccuracy);
			}
			
			avgAccuracy /= numIterations;
			System.out.println("For Training Set Size = " + trainingSetSize[i] + " : Avg Accuracy = " + avgAccuracy);
			System.out.println();
		}
	}
	
}
