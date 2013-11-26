
/**
 * Class to represent a single generalized feature.
 *
 * @author Prakhar Panwaria
 * @date 10/05/2013
 * @hw 1
 */
class Feature 
{
	  private String name;  
	  private int mType;
	  private int mIndex;
	  
	  public static final int TYPE_NUMERIC = 1;
	  public static final int TYPE_DISCRETE = 2;
	  public static final int TYPE_INVALID = -1;
	  
	  public Feature (String name, int index, int type)
	  {
		  this.name = name;
		  this.setIndex(index);
		  this.setType(type);
	  }
		
	  public String getName()
	  {
	    	return name;
	  }

	  public int getType()
	  {
		  return mType;
	  }
	
	  public void setType(int type)
	  {
		  mType = type;
	  }

	  public int getIndex()
	  {
		  return mIndex;
	  }
	  
	  public void setIndex(int index)
	  {
		  mIndex = index;
	  }	  
}

