/**
 * Class to have utility functions and constants.
 * 
 * @author Prakhar
 * @date 11/22/2013
 * @hw 3
 */
public class Utility
{
	public static final boolean IS_VERBOSE = false;	// For debugging purpose.

	/**
	 * Method to get small random data set out of existing data sets.
	 * @param d
	 * @param reqdSetSize
	 * @return
	 */
	public static DataSet getRandomDataSet(DataSet d, int reqdSetSize)
	{
		int dataSetSize = d.size();
		DataSet newDataSet = new DataSet(d); // Getting other dataset properties as well.
		
		// Pick random examples from the given dataset
		for (int i = 0; i < reqdSetSize; i++)
		{
			int index = (int)(Math.random() * dataSetSize);
			newDataSet.add(d.get(index));			
		}
		
		return newDataSet;
	}
}
