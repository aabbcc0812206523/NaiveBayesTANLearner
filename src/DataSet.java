import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Class to represent set of examples. It can be any kind of data 
 * set - training data set, test data set, etc.
 * 
 * @author Prakhar
 * @date 11/22/2013
 * @hw 3
 */

class DataSet extends ArrayList<Example>
{
	private static final long serialVersionUID = 1L;

	// The name of the dataset.
	private String mDataSetName = "";
	
	// Name of relation
	private String mRelationName = "";
	// ArrayList of Features
	private ArrayList<Feature> mFeatures = new ArrayList<Feature>();

	// The number of mFeaturesArray per example in the dataset.
	private int mNumFeatures = 0;

	public DataSet() {}
	
	public DataSet(DataSet d)
	{
		this.mNumFeatures = d.mNumFeatures;
		this.mFeatures = d.mFeatures;
		this.mDataSetName = d.mDataSetName;
		this.mRelationName = d.mRelationName;
	}
	
	public ArrayList<Double> FindCandidateSplits(NumericFeature f)
	{
		if(size() == 0) return null;
		
		ArrayList<Double> splitList = new ArrayList<Double>();
		ArrayList<Example> sortedExampleList = new ArrayList<Example>(this);
		
		final int featureIndex = f.getIndex();
		Collections.sort(sortedExampleList, new Comparator<Example>()
		{
			@Override
			public int compare(Example e1, Example e2)
			{
				return (e1.get(featureIndex).compareTo(e2.get(featureIndex)));
			}
		});

		// Mapping feature values with a group label, which will be marked
		// as 'null' in case there is a mix of labels.
		int outputIndex = getOutputIndex();
		LinkedHashMap<String, String> groupLabelMap = new LinkedHashMap<String, String>();
		for (Example e : sortedExampleList)
		{
			String curFeatureValue = e.get(featureIndex);
			if(groupLabelMap.containsKey(curFeatureValue))
			{
				String storedGroupLabel = groupLabelMap.get(curFeatureValue);
				if(!storedGroupLabel.equals(e.get(outputIndex)))
				{
					groupLabelMap.put(curFeatureValue, "null");
				}
			}
			else
				groupLabelMap.put(curFeatureValue, e.get(outputIndex));
		}
		
		int i = 0;
		String prevGroupLabel = "";
		Double prevVal = 0.0;
		for (Map.Entry<String, String> entry : groupLabelMap.entrySet())
		{
			if(i == 0)
			{
				prevVal = Double.parseDouble(entry.getKey());
				prevGroupLabel = entry.getValue();
				i++;
				continue;
			}

			Double curVal =  Double.parseDouble(entry.getKey());
			String curGroupLabel = entry.getValue();
			
			if(prevGroupLabel.equals("null") || !curGroupLabel.equals(prevGroupLabel))
			{
				// Calculate threshold as the mid point.
				Double threshold = (prevVal + curVal)/2;
				splitList.add(threshold);
			}
			
			prevVal = curVal;
			prevGroupLabel = curGroupLabel;
			
			i++;
		}
		
		return splitList;
	}
	
	int getOutputIndex()
	{
		return mNumFeatures - 1;
	}
	
	public DiscreteFeature getOutputFeature()
	{
		return (DiscreteFeature) mFeatures.get(getOutputIndex());
	}
	
	public int getCountOfExamplesWithGivenOutputValue(String value)
	{
		int index = getOutputIndex();
		int count = 0;
		
		for (int i = 0; i < size(); i++)
		{
			if(this.get(i).get(index).equals(value))
				count++;				
		}
		
		return count;
	}

	/**
	 * Method to find which is the majority label out of all examples.
	 * 
	 * @param outputLabel
	 *            Output Label
	 * @return Majority Label
	 */
	public String MajorityValue()
	{
		DiscreteFeature outputLabel = getOutputFeature();
		String majValue = "";
		int count, maxCount = -1;
		
		for (String value : outputLabel.getValues())
		{
			count = getCountOfExamplesWithGivenOutputValue(value);
	    	
			if(count > maxCount)
			{
				maxCount = count;
				majValue = value;
			}
		}
		
		return majValue;
	}

	/**
	 * Method to check whether all the examples have same classification.
	 * 
	 * @param outputLabel
	 *            Output Label
	 * @return Label, if it is all over the examples, else null.
	 */
	public String isSameClassification()
	{
		int numExamples = size();
		if (numExamples == 0) return null;
		
		Feature f = getOutputFeature();
		
		for (String value : ((DiscreteFeature)f).getValues())
		{
			int count = getCountOfExamplesWithGivenOutputValue(value);
	    	
			if(count < numExamples && count > 0)
				return null;
			else if(count == numExamples)
				return value;
		}
		
		return null;
	}
	
	// Print out a high-level description of the dataset including its mFeaturesArray.
	public void DescribeDataset()
	{
		System.out.println( "=====================================================================" +
							"\nDataset File:\t'" + mDataSetName + "' " +
							"\nRelation Name:\t '" + mRelationName + "' " +
							"\nDataset contains " + size() + " examples, each with " + mNumFeatures + " features.");
		
		System.out.println("The feature names (with their possible values) are:");
		for (int i = 0; i < mNumFeatures; i++)
		{
			Feature f = mFeatures.get(i);
			if(f.getType() == Feature.TYPE_NUMERIC)
			{
				System.out.println("\t" + f.getName().toUpperCase() + ":\t[Real Valued Feature]");
			}
			else
			{
				DiscreteFeature temp = (DiscreteFeature)f;
				System.out.print("\t" + f.getName().toUpperCase() + ":\t");
				for(String values : temp.getValues())
					System.out.print(values + ", ");
				System.out.println();
			}
		}
		System.out.println("=====================================================================");
	}

	// Print out ALL the examples.
	public void PrintAllExamples()
	{
		System.out.println("List of Examples\n================");
		for (int i = 0; i < size(); i++)
		{
			Example thisExample = this.get(i);
			thisExample.PrintFeatures();
		}
	}

	// Returns the number of mFeaturesArray in the data.
	public int getNumberOfFeatures()
	{
		return mNumFeatures;
	}

	public ArrayList<Feature> getFeatures()
	{
		return mFeatures;
	}

	// Returns the name of the ith feature.
	public String getFeatureName(int i)
	{
		return mFeatures.get(i).getName();
	}

	// Parsing data set and storing in relevant data struvtures. 
	public boolean ReadInExamplesFromFile(String dataFile)
	{
		mDataSetName = dataFile;

		// Try creating a scanner to read the input file.
		Scanner fileScanner = null;
		
		try { fileScanner = new Scanner(new File(dataFile)); }
		catch (FileNotFoundException e)	{ return false; }

		// If the file was successfully opened, read the file
		this.parse(fileScanner);
		
		return true;
	}

	/**
	 * Does the actual parsing work. We assume that the file is in proper
	 * format.
	 * 
	 * @param fileScanner  a Scanner which has been successfully opened to read the dataset file
	 */
	public void parse(Scanner fileScanner)
	{
		boolean parseExamplesNow = false;
		int examplesCount = 0;
		
		// Parse whole file
		while (fileScanner.hasNextLine())
		{
			String line = fileScanner.nextLine().trim();
			
			if (!isLineSignificant(line)) continue;
			
			if(parseExamplesNow)	// Examples
			{
				String parts[] = line.split(",");
				
				if(parts.length != mNumFeatures)
				{
					System.out.println(mNumFeatures);
					System.err.println("Error: Ignoring bad Instance (having less feature values).");
					continue;
				}
				
				Example e = new Example(this, examplesCount);
				for (String part : parts)
				{
					e.addFeatureValue(part);
				}
				this.add(e);
				
				examplesCount++;
			}
			else if(line.startsWith("@relation"))	// Relation Name
			{
				String parts[] = line.split(" ");
				
				if(parts[1] != null)
					mRelationName = parts[1];
			}
			else if (line.startsWith("@attribute"))	// Features
			{
				String[] parts = line.split("[,'{}\\s+]");
				
				ArrayList<String> keyParts = new ArrayList<String>();
				for (String part : parts)
				{
					if(part.trim().length() == 0) continue;
					keyParts.add(part.trim());
				}
				
				int keyPartsLength = keyParts.size();
				if(keyPartsLength < 3) continue;
				
				Feature f = null;	
				String fName = keyParts.get(1);
				if(keyParts.get(2).equals("real"))	// Numeric Feature
				{
					f = new NumericFeature(fName, mNumFeatures);
				}
				else
				{
					ArrayList<String> values = new ArrayList<String>();
					for (int i = 2; i < keyPartsLength; i++)
						values.add(keyParts.get(i));
					
					f = new DiscreteFeature(fName, mNumFeatures, values.size(), values);
				}
				
				mFeatures.add(f);
				mNumFeatures++;
			}
			else if (line.startsWith("@data"))		// Examples
			{
				parseExamplesNow = true;
			}
			
		}
	}

	/**
	 * Returns whether the given line is significant (i.e., not blank or a
	 * comment). The line should be trimmed before calling this.
	 * 
	 * @param line line to check
	 */
	private boolean isLineSignificant(String line)
	{
		// Blank lines are not significant. And, lines which have consecutive forward slashes as their 
		// first two characters are comments and are not significant.
		if (line.length() == 0 || (line.length() > 2 && line.substring(0, 2).equals("//")))	return false;

		return true;
	}
	
	public DataSet[] ExamplesForNumericFeatureBranches(int featureIndex, Double threshold)
	{
		Feature f = mFeatures.get(featureIndex);
		if(threshold == null || f == null || f.getType() != Feature.TYPE_NUMERIC) return null;
		
		DataSet[] dataSetArray = new DataSet[2]; // DataSet[0] : Less-Than-EqualTo Examples, and DataSet[1] : More than Examples
		for (int i = 0; i < dataSetArray.length; i++)
			dataSetArray[i] = new DataSet(this);
		
		for (int i = 0; i < size(); i++)
		{
			Example e = get(i);
			
			Double value = Double.parseDouble(e.get(featureIndex));
			
			if (value.compareTo(threshold) <= 0)
				dataSetArray[0].add(e);
			else
				dataSetArray[1].add(e);
		}
		
		return dataSetArray;
	}
	
	public DataSet[] ExamplesForDiscreteFeatureBranches(int featureIndex)
	{
		Feature f = mFeatures.get(featureIndex);
		
		if(f == null || f.getType() != Feature.TYPE_DISCRETE) return null;
		
		DiscreteFeature df = (DiscreteFeature) f;
		int numValues = df.getNumValues();
		
		DataSet[] dataSetArray = new DataSet[numValues]; // Getting examples for all the branches.
		for (int i = 0; i < dataSetArray.length; i++)
			dataSetArray[i] = new DataSet(this);
		
		for (int i = 0; i < size(); i++)
		{
			Example e = get(i);
			String value = e.get(featureIndex);
			
			for (int j = 0; j < numValues; j++)
			{
				if(value.equals(df.getValues().get(j)))
					dataSetArray[j].add(e);
			}
		}
		
		return dataSetArray;
	}
	
	public int[] getBreakUpOfOutputValues()
	{
		Feature f = getOutputFeature();
		int numOutputValues = ((DiscreteFeature)f).getNumValues();
		
		int[] countArray = new int [numOutputValues];
		for (int i = 0; i < numOutputValues; i++)
			countArray[i] = 0;
		
		int numExamples = size();
		if (numExamples == 0) return countArray;
		
		int i = 0;
		for (String value : ((DiscreteFeature)f).getValues())
		{
			countArray[i++] = getCountOfExamplesWithGivenOutputValue(value);
		}
		
		return countArray;
	}

}
