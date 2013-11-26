import java.util.ArrayList;

/**
 * Class to represent an example from the data set.
 * 
 * @author Prakhar
 * @hw 1
 */

public class Example extends ArrayList<String>
{
	private static final long serialVersionUID = 1L;

	// The index of this example.
	private int index;

	// The output label of this example.
	private String label;

	// The data set in which this is one example.
	public DataSet parent;

	// Constructor which stores the dataset which the example belongs to.
	public Example(DataSet parent, int index)
	{
		this.parent = parent;
		this.index = index;
	}

	// Print out this example in human-readable form.
	public void PrintFeatures()
	{
		System.out.print("Example " + index + ",  label = " + label + "\n");
		for (int i = 0; i < parent.getNumberOfFeatures(); i++)
		{
			System.out.print("     " + parent.getFeatureName(i) + " = "
					+ this.get(i) + "\n");
		}
	}

	// Adds a feature value to the example.
	public void addFeatureValue(String value)
	{
		this.add(value);
	}

	public int getName()
	{
		return index;
	}

	public String getLabel()
	{
		return get(parent.getOutputIndex());
	}

	public void setIndex(int i)
	{
		this.index = i;
	}

	public void setLabel(String label)
	{
		this.label = label;
	}
}
