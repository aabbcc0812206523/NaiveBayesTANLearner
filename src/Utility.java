
public class Utility
{
	
	public static final boolean IS_VERBOSE = false;

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
