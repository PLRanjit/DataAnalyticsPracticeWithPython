{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\A638127\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\matplotlib\\axes\\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.\n",
      "  warnings.warn(\"The 'normed' kwarg is deprecated, and has been \"\n",
      "C:\\Users\\A638127\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\matplotlib\\axes\\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.\n",
      "  warnings.warn(\"The 'normed' kwarg is deprecated, and has been \"\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.JointGrid at 0x9dd52b0>"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Jz+ZAw44ID/7sz+b9773vTnrrLNy//3356CDDspf/dVf5Stf+Uq++tWv5s4778yGDRvykz/5k0mSL3/5y7nxxhtz9NFH56yzzspnP/vZPOc5z0kyuZXeP/3TP+019w996EP5gz/4g73aTzjhhFx99dV7tH3ve9/Lscc+utpz9erVueGGG2b8XP7t3/4tt9xyS57//OcnSX74wx/mv/7X/5q//Mu/zKc+9ak9jv2Lv/iLGfvZvHlzrr/++nznO9/J8573vNx888056KCDpl6/77778hM/8RN9z73iiiv2quBfc801OeaYY7Ju3boZx3z2s5+d5z3veVm1alVqrXnta1+bk08+OXfeeWfe9ra35ZOf/GQOPvjg/P7v/37+6I/+KL/7u7+7Vx8PPvhg1q9fnxUrVuSSSy7Jz/3cz804HvQa9ZIVAFj0jj322Jx11llJkl/8xV/Mu9/97px77rn5+te/nnPOOSfJ5HKPVatWJUm+/vWv581vfnPuvvvu3H///XnhCx+9N90FF1wwtaTkrLPOyhve8IZcdNFF+fmf//msXr06//RP/5SXv/zlOeCAA/KkJz0pz33uc7Np06YccsghedaznpXVq1cnSU4//fRs3rx5KpD/wi/8Qt+5X3TRRbnooovm9D77raGe7QLEq666Kueff/7U+3nf+96Xn/7pn94j1M/FS1/60jzmMY/JiSeemOOPPz7f/OY3c/rpp0+9/oQnPKHv0pB+Hnjggbz97W/PtddeO+txN998c77xjW9k69atSZJzzjknn/nMZ3Lvvffmpptumvp579q1K89+9rP79nHrrbfm6KOPzne/+908//nPz9Of/vQ8+clPntM8Wd4EcgCYp+mhtJSSWmtOPfXUfP7ze9/e4pWvfGX+5m/+JuvWrctll12WT3/601OvHXzwwVOPL7nkkvzMz/xMPvaxj+XMM8/MJz/5yVkvLHzc4x439fiAAw7Iww8/PPW8t99e86mQr169Olu2PLpj8NatW3P00UfPOJ+rrroq733ve6eef/7zn88//uM/5n3ve1/uv//+7Nq1Kz/yIz8y48WhXf0+317zqZB/5zvfyS233DJVHd+6dWue8Yxn5Itf/GJ+7Md+bOq4v/7rv86ZZ56ZH/mRH0mSvOhFL8oXvvCFnHzyyTnnnHNy5ZVX7jHODTfckF/7tV9Lkrz1rW/NeeedN/XZHH/88Tn77LPz5S9/WSBnTka9ywoALHq33nrrVPC+8sor85znPCcnnXRS7rjjjqn23bt358YbJy97uu+++7Jq1ars3r07H/rQh2bs9zvf+U6e/vSn501velPWr1+fb37zm/nJn/zJfPjDH84jjzySO+64I5/5zGfyrGfttWJzzi666KJ85Stf2evP9DCeJBs2bMi3v/3t3HLLLdm1a1euuuqqPXYP6fWtb30r27dv36N6/KEPfSi33nprNm/enD/8wz/MK17xiqkw/opXvCJf/GL/vRU2btyYH/7wh/nOd76T7373uznppJP2eL1bIe/3Z/pylac//en5wQ9+kM2bN2fz5s1ZvXp1/vmf/3mPMJ4ka9asyfXXX5+HH344u3fvzvXXX5+TTz45Z555Zj772c/m5ptvTjJZcf/Xf/3XnHHGGVNjnnfeedm+fXseeuihJMmdd96Zz372s3vNBWYikAPAPJ188sm5/PLLc9ppp2Xbtm35jd/4jRx44IG5+uqr86Y3vSnr1q3L6aefns997nNJkt/7vd/LGWeckXPOOSdPfepTZ+z3j//4j/O0pz0t69aty8qVK/OiF70o//E//secdtppWbduXZ7//Ofnne98515hclhWrFiR97znPXnhC1+Yk08+OS996Utz6qmnJkl+93d/N9dcc83UsVdeeWVe9rKXzXlP7a997WtTS3qmO+mkk/Lc5z43L3rRi/Jnf/Zne6wfH6SJiYn8yq/8SpLk/PPPz5Of/OQ8/elPz7p167Ju3br87M/+bI466qhcdtllefnLX57TTjstZ555Zr75zW/u1dc3vvGNrF+/PuvWrcvznve8XHLJJVOB/N3vfndWr16drVu35rTTTpsaE7rKfPfYXG7Wr19f+10FDkAb3/jGN3LyySc3G3/z5s158YtfnK9//evN5rDY3XvvvXnVq17VdweYV77ylXnxi1+c888/v8HM2pnh3/XA7hgkzzQzp5+hCjkAMFKHHHLIfm3HCEuNizoBYB7Wrl2rOj5El112WespwMipkAMAQEMCOQCLjuufWEr8e0YgB2BROeigg3LXXXcJMTS1bceubNuxa7/7qbXmrrvuGtpOMiwO1pADsKh0t4+74447Wk+FZWzHQ5M3Yfr+4/Y/Sh100EFTd1xleRLIAVhUHvvYx+a4445rPQ2WqY0Tk3cu3bp9Z5Jk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\n",
      "text/plain": [
       "<Figure size 720x720 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd  # data handeling\n",
    "import numpy as np   # numeriacal computing\n",
    "import matplotlib.pyplot as plt  # plotting core\n",
    "import seaborn as sns  # higher level plotting tools\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "pd.set_option('display.float_format', lambda x: '%.2f' % x)\n",
    "pd.set_option('display.max_columns', 21)\n",
    "pd.set_option('display.max_rows', 70)\n",
    "\n",
    "df = pd.read_csv(\"kc_house_data.csv\")  # create a dataframe with pandas \"pd\"\n",
    "\n",
    "df.head()  # display the first few lines of the dataframe \"df\"\n",
    "\n",
    "df.isnull().values.any()  # check for missing values\n",
    "\n",
    "#df[[\"price\",\"bedrooms\",\"bathrooms\",\"sqft_living\",\"sqft_lot\",\"sqft_above\",\"yr_built\",\"sqft_living15\",\"sqft_lot15\"]].describe()\n",
    "\n",
    "#sns.pairplot(data=df, x_vars=['sqft_living','sqft_lot','sqft_above','sqft_living15','sqft_lot15'], y_vars=[\"price\"])\n",
    "\n",
    "df2 = df[[\"price\", \"sqft_living\"]]\n",
    "df2.head()\n",
    "\n",
    "#sns.jointplot('sqft_living','price', data=df2, size=10, alpha=.5, marker='+')\n",
    "#df.groupby(['zipcode'])['price'].mean()\n",
    "\n",
    "zip98103 = df['zipcode'] == 98103  # True if zip is 98103\n",
    "zip98039 = df['zipcode'] == 98039\n",
    "\n",
    "#sns.jointplot('sqft_living','price', data=df2[zip98103], size=10, alpha=.5, marker='+')\n",
    "\n",
    "#sns.jointplot('sqft_living','price', data=df2[zip98039], size=10, alpha=.5, marker='+')\n",
    "\n",
    "sns.jointplot('sqft_living','price', data=df2[(df['bedrooms']==3) & (df['zipcode'] == 98103)], size=10, alpha=.5, marker='+')\n",
    "\n",
    "#plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}