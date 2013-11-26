import java.util.ArrayList;
import java.util.HashMap;

/**
 * Class to represent a Discrete Feature.
 * 
 * @author Prakhar Panwaria
 * @date 11/22/2013
 * @hw 3
 */

class DiscreteFeature extends Feature
{
	/* Member Variables */
	
	private int mNumValues;
	private ArrayList<String> mValues = null;
	public HashMap<String, Integer> valueIndexMap = new HashMap<String, Integer>();
	
	/* Methods */
	
	public DiscreteFeature(String name, int index, int numValues, ArrayList<String> values)
	{
		super(name, index, Feature.TYPE_DISCRETE);
		mNumValues = numValues;
		mValues = values;
		
		int i = 0;
		for(String v : mValues)
			valueIndexMap.put(v, i++);
	}
	
	public int getNumValues()
	{
		return mNumValues;
	}

	public void setNumValues(int numValues)
	{
		this.mNumValues = numValues;
	}

	public ArrayList<String> getValues()
	{
		return mValues;
	}

	public void setValues(ArrayList<String> values)
	{
		this.mValues = values;
	}

}
