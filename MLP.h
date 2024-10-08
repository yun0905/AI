#pragma once
#define LEARNING_RATE0
class CMLP
{
public:
	CMLP();
	~CMLP();

	// �Է³���� ��
	int m_iNuminNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;   //hidden only
	int m_iNumTotalLayer;    // inputlayer + hiddenlayer + outputlayer
	int* m_NumNodes;         // [0] - input node, [1.] - hidden layer, [m.iNumHiddenLayer + 1], output layer

	double*** m_Weight;      // [����layer][���۳��][������]
	double** m_NodeOut;      // [layer][node]

	double* pInValue, * pOutValue;   // �Է·��̾�, ��·��̾�
	double* pCorrectOutValue;        // ���䷹�̾�

	double** m_ErrorGradient; // [layer][node]


	bool Create(int InNode, int* pHiddenNode, int OutNode, int NumHiddenLayer);
private:
	void InitW();
	double ActivationFunc(double weightsum);
public:
	void Forward();
	void BackPopagationLearning();
};

