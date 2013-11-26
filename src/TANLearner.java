import java.util.ArrayList;
import java.util.HashSet;

public class TANLearner extends Learner
{

	public TANLearner(DataSet trainSet)
	{
		super(trainSet);
	}
	
	/**
	 * Main method for TAN Learner to learn the Bayes Net Structure.
	 */
	public void learnBayesNetStructure()
	{
		// Prepare the basic model.
		prepareBasicModel();	// It will prepare mBasicModelParams
		
		// Prepare a data structure according to the training set
		// which can help in calculating the weight of the edges of
		// the graph calculated in next step.
		initComplexModelParams();
		prepareComplexModelParams();
		printComplexModelParams();
		
		// Learn BayesNet Structure -- Taking the first attribute as the root, assign directions
		// in the tree, as well as make Y as parent of all the nodes in the graph.
		getBayesNetStructure();
		printBayesNetStructure();
		
		// Calculate CPT Tables for each node in the learned Bayes Net
		for (int i = 0; i < mNumSoleFeatures; i++)
			mCPTable.add(calculateCPTForFeature(i));
		
	}
	
	/**
	 * Method to initialize Complex Model Parameters Matrix.
	 */
	public void initComplexModelParams()
	{
		mComplexModelParams = new Integer[mNumSoleFeatures][mNumSoleFeatures][][][];

		for(int i = 0; i < mNumSoleFeatures; i++)
		{
			int numFeatureValuesi = ((DiscreteFeature)mFeatures.get(i)).getNumValues();
			
			for(int j = i+1; j < mNumSoleFeatures; j++)
			{
				int numFeatureValuesj = ((DiscreteFeature)mFeatures.get(j)).getNumValues();
				mComplexModelParams[i][j] = new Integer[numFeatureValuesi][numFeatureValuesj][];
				
				for(int k = 0; k < numFeatureValuesi; k++)
					for(int m = 0; m < numFeatureValuesj; m++)
					{
						mComplexModelParams[i][j][k][m] = new Integer[2];	// Y is binary-valued.
						mComplexModelParams[i][j][k][m][0] = 0;
						mComplexModelParams[i][j][k][m][1] = 0;
					}
			}
		}
	}
	
	/**
	 * Method to prepare Complex Model Parameters Matrix.
	 */
	public void prepareComplexModelParams()
	{
		String classFirstValue = mTrainSet.getOutputFeature().getValues().get(0);
		
		// Run through the examples now.
		for(Example e : mTrainSet)
		{
			// ASSUMPTION: Class label is binary-valued.
			int classLabelIndex = e.get(mNumFeatures - 1).equals(classFirstValue) ? 0 : 1;
			
			// Go through all the feature values of this training
			// set instance and update conditional probability tables.
			for(int i = 0; i < mNumSoleFeatures; i++)
			{
				DiscreteFeature featurei = (DiscreteFeature) mFeatures.get(i);
				int featureValueIndexi =  featurei.valueIndexMap.get(e.get(i));	// Locate the feature value
				
				for(int j = i+1; j < mNumSoleFeatures; j++)
				{
					DiscreteFeature featurej = (DiscreteFeature) mFeatures.get(j);
					int featureValueIndexj =  featurej.valueIndexMap.get(e.get(j));	// Locate the feature value
					
					// Updating the count
					mComplexModelParams[i][j][featureValueIndexi][featureValueIndexj][classLabelIndex]++;
				}
			}
		}
	}
	
	/**
	 * Method to print Complex Model Parameter values.
	 */
	public void printComplexModelParams()
	{
		if(!Utility.IS_VERBOSE) return;
		
		for(int i = 0; i < mNumSoleFeatures; i++)
		{
			int numFeatureValuesi = ((DiscreteFeature)mFeatures.get(i)).getNumValues();
			
			for(int j = i+1; j < mNumSoleFeatures; j++)
			{
				System.out.println("\nFor [" + i + "][" + j + "]\n---------------");
				
				int numFeatureValuesj = ((DiscreteFeature)mFeatures.get(j)).getNumValues();
				for(int k = 0; k < numFeatureValuesi; k++)		// Feature-i values
				{
					for(int m = 0; m < numFeatureValuesj; m++)	// Feature-j values
					{
						System.out.print("<" + mComplexModelParams[i][j][k][m][0] + "::" + mComplexModelParams[i][j][k][m][0] + "> ");
					}
					
					System.out.println();
				}
			}
		}
	}
	
	/**
	 * Method to get Bayes Net Structure by first calculating Weighted Graph and then
	 * getting Maximim Spanning Tree from it using Prim's Algorithm.
	 */
	public void getBayesNetStructure()
	{
		// Prepare complete graph between different features
		calculateGraphEdges();
		printWeightedGraph();
		
		// Use Prim's Algorithm to get MST out of that graph
		getMST();
	}
	
	/**
	 * Method to calculate weight of the graph edges, which will be 
	 * then used to find the structure of Bayes Net.
	 */
	public void calculateGraphEdges()
	{
		mWeightedGraph = new Double[mNumSoleFeatures][mNumSoleFeatures];
		
		for(int i = 0; i < mNumSoleFeatures; i++)
			for(int j = 0; j < mNumSoleFeatures; j++)
				mWeightedGraph[i][j] = -1.0;
				
		for(int i = 0; i < mNumSoleFeatures; i++)
		{
			int numFeatureValuesi = ((DiscreteFeature)mFeatures.get(i)).getNumValues();
			
			for(int j = i+1; j < mNumSoleFeatures; j++)
			{
				int numFeatureValuesj = ((DiscreteFeature)mFeatures.get(j)).getNumValues();
				
				double P_Xi_Xj_given_Y = 0.0;
						
				for(int k = 0; k < numFeatureValuesi; k++)		// Feature-i values
				{
					for(int m = 0; m < numFeatureValuesj; m++)	// Feature-j values
					{
						for(int n = 0; n < 2; n++)	// Output Y values
						{
							double P_xi_xj_y = 		 ((double) (mComplexModelParams[i][j][k][m][n] + 1)) / (mTrainSet.size() + (numFeatureValuesi * numFeatureValuesj * 2));
							double P_xi_xj_given_y = ((double) (mComplexModelParams[i][j][k][m][n] + 1)) / (mOutputValueCount[n] + (numFeatureValuesi * numFeatureValuesj));
							double P_xi_given_y =    ((double) (mBasicModelParams.get(i).get(k)[n] + 1)) / (mOutputValueCount[n] + numFeatureValuesi);
							double P_xj_given_y =    ((double) (mBasicModelParams.get(j).get(m)[n] + 1)) / (mOutputValueCount[n] + numFeatureValuesj);
							
							double weight = P_xi_xj_y * Math.log( P_xi_xj_given_y / (P_xi_given_y * P_xj_given_y) ) / Math.log(2.0);	// TODO: Check if log2?
							
							if(Utility.IS_VERBOSE)
							{
								System.out.println("["+i+"]["+j+"]["+k+"]["+m+"]["+n+"]: "+ 
													P_xi_xj_y + " " + 
													P_xi_xj_given_y + " " +
													P_xi_given_y + " " +
													P_xj_given_y + " Weight=" +
													weight);
							}
							
							P_Xi_Xj_given_Y += weight;
							
						}
					}
				}
				
				// Add the weight to the graph
				mWeightedGraph[i][j] = P_Xi_Xj_given_Y;
				mWeightedGraph[j][i] = P_Xi_Xj_given_Y;
				
			}
		}
	}
	
	/**
	 * Method to print the Weighted Graph.
	 */
	void printWeightedGraph()
	{
		if(!Utility.IS_VERBOSE) return;
		
		System.out.println("\nPrinting WEIGHTED GRAPH\n-----------------------");
		
		for (int i = 0; i < mNumSoleFeatures; i++)
		{
			for(int j = 0; j < mNumSoleFeatures; j++)
			{
				Double number = (double) Math.round(mWeightedGraph[i][j] * 10000);
				number = number/10000;
//					System.out.print(number + "\t");
				System.out.print(mWeightedGraph[i][j] + "\t");
			}
			System.out.println();
		}
	}
	
	/**
	 * Method to get Maximum Spanning Tree from the Weighted Graph using Prim's Algorithm.
	 * @return	Maximum Spanning Tree
	 */
	Integer[] getMST() { return getMST(mWeightedGraph); }
	Integer[] getMST(Double[][] graph)
	{
		HashSet<Integer> visitedSet = new HashSet<Integer>();
		mBayesNetStructure = new Integer[mNumSoleFeatures];
		
		// Pushing first feature in the Visited Node set.
		visitedSet.add(0);
		int numVisited = 1;
		mBayesNetStructure[0] = -1;	// Having no parent, hence, this will be the root of the tree.
		for (int i = 0; i < mNumSoleFeatures; i++)
			mWeightedGraph[i][0] = -1.0;
		
		// Iterate until every node is visited.
		while(numVisited < mNumSoleFeatures)
		{
			Double tempMax = -1.0;
			int sourceEdge = -1;
			int destEdge = -1;
			
			// Only considering so far visited nodes
			for (Integer i : visitedSet) 
			{
				// Finding max-weighted edge for this feature
			    for(int j = 0; j < mNumSoleFeatures; j++)
			    {
			    	if(mWeightedGraph[i][j].compareTo(-1.0) == 0) continue;
			    	
			    	if( (tempMax.compareTo(mWeightedGraph[i][j]) == 0 && i < sourceEdge) ||
			    		(tempMax.compareTo(mWeightedGraph[i][j]) < 0))
			    	{
			    		tempMax = mWeightedGraph[i][j];
			    		sourceEdge = i;
			    		destEdge = j;
			    	}
			    }
			}
			
			if(tempMax.compareTo(-1.0) == 0)	// All nodes have been visited.
				break;
			
			// Add the max-weighted edge in the parent[]
			mBayesNetStructure[destEdge] = sourceEdge;
			
			// Add the 'j' node in the visited sets
			visitedSet.add(destEdge);
			
			if(Utility.IS_VERBOSE)
			{
				System.out.println("\nSourceEdge = [" + sourceEdge + "] DestEdge = [" + destEdge + "]");
				System.out.print("Visited Set = {");
				for (Integer i : visitedSet) System.out.print(i + ", ");
				System.out.print("}\n");
			}
			
			// Remove the column values for the max-weighted j
			for (int i = 0; i < mNumSoleFeatures; i++)
				mWeightedGraph[i][destEdge] = -1.0;
			
			numVisited++;
		}
		
		return mBayesNetStructure;
	}
	
	/**
	 * Method to print Bayes Net.
	 */
	void printBayesNetStructure()
	{
		for (int i = 0; i < mBayesNetStructure.length; i++)
		{
			if(Utility.IS_VERBOSE)
				System.out.print( "(" + mBayesNetStructure[i] + ", " + i + ") , ");
			
			if(mBayesNetStructure[i] == -1)
				System.out.println(mFeatures.get(i).getName() + " class");
			else
				System.out.println(mFeatures.get(i).getName() + " " + mFeatures.get(mBayesNetStructure[i]).getName() + " class");
		}
		
		System.out.println();
	}
	
	/**
	 * Method to calculate Conditional Probability Table for each feature in the Bayes Net.
	 * @param i		Feature Index
	 * @return		CP Table for this feature
	 */
	Double[][][] calculateCPTForFeature(int i)
	{
		if(Utility.IS_VERBOSE)
		{
			System.out.println("\n\nPrinting CPT for Feature[" + i + "]\n---------------------------------");
			System.out.println("\nCPT of Attribute " + i);
		}
		
		Double[][][] CPT = null;
		DiscreteFeature f = (DiscreteFeature)mFeatures.get(i);
		int numFeatureValues = f.getNumValues();
		
		if(mBayesNetStructure[i] == -1)
		{
			CPT = new Double[1][numFeatureValues][2];
			
			if(Utility.IS_VERBOSE)
				System.out.println("For Feature="+ i + "\tParent=" + mTrainSet.getOutputIndex());
			
			for(int k = 0; k < numFeatureValues; k++)
			{	
				for(int y = 0; y < 2; y++)
				{
					double num = ((double) (mBasicModelParams.get(i).get(k)[y] + 1));
					double den = (mOutputValueCount[y] + numFeatureValues);
					
					CPT[0][k][y] = ((double) num) / den;
					
					if(Utility.IS_VERBOSE)
						System.out.println("Pr(" + i + "=" + k + " | 18=" + y + ") = " + CPT[0][k][y]);
				}
			}
		}
		else
		{
			int parenti = mBayesNetStructure[i];
			
			if(Utility.IS_VERBOSE)
				System.out.println("Parent["+ i + "]=" + parenti);
			
			DiscreteFeature parentFeature = (DiscreteFeature)mFeatures.get(parenti);
			int parentNumFeatureValues = parentFeature.getNumValues();
			
			CPT = new Double[parentNumFeatureValues][numFeatureValues][2];
			
			for(int m = 0; m < parentNumFeatureValues; m++)
			{	
				for(int k = 0; k < numFeatureValues; k++)
				{
					for(int y = 0; y < 2; y++)
					{
						int x1 = i, x2 = parenti;
						int index1 = k, index2 = m;
						if(parenti < i)	{ x1 = parenti; index1 = m; x2 = i; index2 = k;}
						
						double num = ((double) (mComplexModelParams[x1][x2][index1][index2][y] + 1)) / (mTrainSet.size() + (numFeatureValues * parentNumFeatureValues * 2));
						double den = ((double) (mBasicModelParams.get(parenti).get(m)[y] + numFeatureValues)) / (mTrainSet.size() + numFeatureValues * parentNumFeatureValues * 2);
						
						CPT[m][k][y] = ((double) num) / den;
						
						if(Utility.IS_VERBOSE)
							System.out.println("Pr(" + i + "=" + k + " | " + parenti + "=" + m + ",18=" + y + ") = " + CPT[m][k][y]);
					}
				}
			}
			
		}
		
		return CPT;
	}
	
	
	/**
	 * Method to test the Bayes Net Structure generated.
	 * @param testSet	Test Data Set
	 * @return			Test Set Accuracy
	 */
	Double testModel(DataSet testSet)
	{
		int correctPredictionCount = 0;
		ArrayList<String> classValues = mTrainSet.getOutputFeature().getValues();
		
		// Run through the examples now.
		for(Example e : testSet)
		{
			// Initialize the ratios to be calculated
			Double[] ratio = new Double[2];
			for(int y = 0; y < 2; y++)	// ASSUMING Output label is binary-valued.
				ratio[y] = (double)(mOutputValueCount[y] + 1);	
			
			// Go through all the feature values of the example
			for(int i = 0; i < mBayesNetStructure.length; i++)
			{
				// Since we have already calculated CPT Tables for each features, 
				// let's use those to calculate the probability ratio.
				Double[][][] CPT = mCPTable.get(i);
				
				DiscreteFeature f = (DiscreteFeature) mFeatures.get(i);
				int featureValueIndex =  f.valueIndexMap.get(e.get(i));	// Locate the feature value
				
				int parent = mBayesNetStructure[i];
				if(parent == -1)	// If node in the structure has 'class' as only parent.
				{
					for(int y = 0; y < 2; y++)
						ratio[y] *= CPT[0][featureValueIndex][y];
				}
				else
				{
					DiscreteFeature parentFeature = ((DiscreteFeature) mFeatures.get(parent));
					int parentFeatureValueIndex = parentFeature.valueIndexMap.get(e.get(parent));
					
					for(int y = 0; y < 2; y++)
						ratio[y] *= CPT[parentFeatureValueIndex][featureValueIndex][y];
				}
				
				/*******************************************************************************************
				//   ALTERNATIVE WAY -- Calculations without using CPT Table
				//*******************************************************************************************
				if(parent == -1)	// The only parent is the output class
				{
					// Look up in the BasicModelParams
					DiscreteFeature f = (DiscreteFeature) mFeatures.get(i);
					int featureValueIndex =  f.valueIndexMap.get(e.get(i));	// Locate the feature value
					
					ArrayList<Integer[]> featureArray = mBasicModelParams.get(i);
					int numFeatureValues = f.getNumValues();	// Number of values for current feature, used for ensuring Laplace Estimates.
					for(int y = 0; y < 2; y++)	// ASSUMING Output label is binary-valued.
					{
						ratio[y] *= ((double)(featureArray.get(featureValueIndex)[y] + 1)/(mOutputValueCount[y] + numFeatureValues));
					}
				}
				else
				{
					int indexXi = i, indexXj = parent;
					if(parent < i) { indexXi = parent; indexXj = i; }
					int numFeatureValuesI = ((DiscreteFeature) mFeatures.get(i)).getNumValues();
					
					DiscreteFeature featureXi = ((DiscreteFeature) mFeatures.get(indexXi));
					DiscreteFeature featureXj = ((DiscreteFeature) mFeatures.get(indexXj));
					int featureValueIndexXi = featureXi.valueIndexMap.get(e.get(indexXi));
					int featureValueIndexXj = featureXj.valueIndexMap.get(e.get(indexXj));
					
					DiscreteFeature parentFeature = ((DiscreteFeature) mFeatures.get(parent));
					int parentFeatureValueIndex = parentFeature.valueIndexMap.get(e.get(parent));	// Locate the feature value
					ArrayList<Integer[]> parentFeatureArray = mBasicModelParams.get(parent);
					
					for(int y = 0; y < 2; y++)	// ASSUMING Output label is binary-valued.
					{
						double P_xi_xj_y = (double) (mComplexModelParams[indexXi][indexXj][featureValueIndexXi][featureValueIndexXj][y] + 1);
						double P_xj_y = (double) (parentFeatureArray.get(parentFeatureValueIndex)[y] + numFeatureValuesI);
						
						ratio[y] *= P_xi_xj_y / P_xj_y;
					}					
				}
				//***********************************************************************************/
				
			}
			
			// Predict the output
			String predictedLabel = "";
			double posteriorProb = 0.0;
			Double numerator = ratio[0], denominator = ratio[1];
			if(numerator.compareTo(denominator) >= 0)	// Predicted Label = FirstLabel
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
	
	/* Member Variables */
	
	// 5-D Array to hold the data from the training set. It helps in creating the weighted graph and also the CP tables.
	// Description - [Feature#i][Feature#j][Feature-i-value][Feature-j-value][y-label]
	public Integer[][][][][] mComplexModelParams = null;
	
	// Complete Graph with all the features as nodes, and having edge values as 
	// Mutual Conditional Information between two features.
	public Double[][] mWeightedGraph = null;
	
	// Bayes Net Structure -- Mapping of node to its parent node
	public Integer[] mBayesNetStructure = null;
	
	// Conditional Probability Table for each node in the Bayes Net Structure
	public ArrayList<Double[][][]> mCPTable = new ArrayList<Double[][][]>();
}
