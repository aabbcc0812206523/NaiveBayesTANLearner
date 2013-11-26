
/**
 * Class to represent a Numeric Feature.
 * 
 * @author Prakhar Panwaria
 * @date 10/05/2013
 * @hw 1
 */

class NumericFeature extends Feature
{
	private Double mThreshold;	// Threshold Value on which we'll branch out in Decision Tree.
	
	public NumericFeature(NumericFeature f, Double threshold)
	{
		super(f.getName(), f.getIndex(), f.getType());
		setThreshold(threshold);
	}
	
	public NumericFeature(String name, int index)
	{
		super(name, index, TYPE_NUMERIC);
		
	}

	public Double getThreshold()
	{
		return mThreshold;
	}

	public void setThreshold(Double threshold)
	{
		mThreshold = threshold;
	}
}
