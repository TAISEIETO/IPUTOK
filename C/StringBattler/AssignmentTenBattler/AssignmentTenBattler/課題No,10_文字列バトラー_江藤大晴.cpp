/* ------------------------------------------------------------------------- */
/* �ۑ�No,10_������o�g���[_�]���吰.cpp									 */
/* AssignmentTenBattler														 */
/* ���͕�����ɂ���ăX�e�[�^�X�����܂�^�[�����o�g��						 */
/*																			 */
/* ------------------------------------------------------------------------- */
/*	�ԍ�	�X�V����								���t		����		 */
/* ------------------------------------------------------------------------- */
/*	000000	�V�K�쐬								2023/08/10	�]��  �吰	 */
/* ------------------------------------------------------------------------- */
#define _CRT_SECURE_NO_WARNINGS					/* scanf�Ή�				 */

/* ------------------------------------------------------------------------- */
/* include�t�@�C��															 */
/* ------------------------------------------------------------------------- */
#include<stdio.h>								/* �W�����o�͐���			 */
#include<time.h>								/* ���Ԑ���					 */
#include<stdlib.h>								/* ��{���C�u����			 */
#include <string.h>								/* �����񐧌�				 */

/* ------------------------------------------------------------------------- */
/* �\���̒�`																 */
/* ------------------------------------------------------------------------- */
struct PLAYER {
	int HP;										/* �̗�						 */
	int ATK;									/* �U����					 */
	int DEF;									/* �^						 */
	int LUCK;									/* �h���					 */
	char NAME[43];								/* �v���C���[��				 */
	char WEAPON[41];							/* ����						 */
};

/* ------------------------------------------------------------------------- */
/* �v���g�^�C�v�錾															 */
/* ------------------------------------------------------------------------- */
int Status(struct PLAYER* fInfo, int fCount);
void SPrint(struct PLAYER* fInfo, int fIndex, int fMaxHP);
int Damage(struct PLAYER* fInfo1, struct PLAYER* fInfo2, int* ptr);
void CPrint(int fCritSw);
void DPrint(struct PLAYER* fInfo1, struct PLAYER* fInfo2, int fDamage, int fIndex1, int fIndex2);
void RPrint(struct PLAYER* fInfo1, struct PLAYER* fInfo2);
int PColorPrint(int fIndex);
int HPColorPrint(struct PLAYER* fInfo, int fMaxHP);
void YWall(int* ptr, int fMax);
void XWall(int* ptr, int fMax);
int* START(int* ptr);
void Movement(int* ptr1, int* ptr3);
void iArrayPrint(int* ptr, int fMax);
void iArrayReset(int* ptr, int fMax);
void cArrayReset(char* ptr, int fMax);
void Next();

/* ------------------------------------------------------------------------- */
/* �֐���	: main															 */
/* �@�\��	: �^�[�����o�g��												 */
/* �@�\�T�v : ���͕�����ɂ���ăX�e�[�^�X�����܂�^�[�����o�g��			 */
/* ����		: void	: f����	:												 */
/* �߂�l	: int	: OS�֕Ԃ��l(����0�Œ�)									 */
/* �쐬��	: 2023/08/10		�]��  �吰		�V�K�쐬					 */
/* ------------------------------------------------------------------------- */
int main(void)
{
	/* �ϐ��錾 ------------------------------------------------------------ */
	struct PLAYER Info[2] = {					/* �\���̔z��				 */
	{/* PLAYER�z��0�Ԗ� ---------------------------------------------------- */
		0,										/* HP						 */
		0,										/* ATK						 */
		0,										/* DEF						 */
		0										/* LUCK						 */
	},
	{/* PLAYER�z��1�Ԗ� ---------------------------------------------------- */
		0,										/* HP						 */
		0,										/* ATK						 */
		0,										/* DEF						 */
		0										/* LUCK						 */
	}
	};
	int iMaxHP[2]{ 0,0 };						/* �ő�HP�L�^�z��			 */
	int iDamage[2]{ 0,0 };						/* �^�_���[�W�v�Z�z��		 */
	int iCritSw[2]{ 0,0 };						/* ��S�̈ꌂ����z��		 */
	int iDeadSw = 0;							/* �퓬�s�\����(2P)			 */
	int iCount = 0;								/* �^�[�����J�E���^			 */
	int iPColor = 0;							/* �v���C���[�\���F����		 */
	int iBox[31][31];							/* �f�[�^2�����z��(31*31)	 */
	int* ptr = 0;								/* �|�C���^					 */
	srand(time(NULL));							/* ����������				 */

	/* �����J�n------------------------------------------------------------- */

	/* �\���̔z��̐����[�v ------------------------------------------------ */
	for (int iIndex = 0; iIndex < 2; iIndex++) {
		cArrayReset(&Info[iIndex].NAME[0], 43);/* ���O�z�񏉊��� */
		cArrayReset(&Info[iIndex].WEAPON[0], 41);/* ����z�񏉊��� */
	}
	
	printf("�Ή������R�[�h:SHIFT_JIS(���p1����1Byte�A�S�p1����2Byte)\n");

	/* �\���̔z��̐����[�v ------------------------------------------------ */
	for (int iIndex = 0; iIndex < 2; iIndex++) {
		iPColor = PColorPrint(iIndex);/* �v���C���[�\���F�t�� */
		printf("\n�v���C���[������͂��Ă��������B(10Byte�ȏ�A40byte�ȓ�)\n[ENTER]�œ��͂��m�肷��B\n\n\x1b[%dm%dP:\x1b[39m",iPColor, iIndex + 1);
		iMaxHP[iIndex] = Status(&Info[iIndex], iCount);/* �X�e�[�^�X�U�蕪�� */
	}
	system("cls");
	printf("�X�e�[�^�X���\n\n\n");

	/* �\���̔z��̐����[�v ------------------------------------------------ */
	for (int iIndex = 0; iIndex < 2; iIndex++) {
		SPrint(&Info[iIndex], iIndex, iMaxHP[iIndex]);/* �X�e�[�^�X���\�� */
	}
	Next();
	system("cls");
	iArrayReset(&iBox[0][0], 31 * 31);/* int�^�z�񏉊��� */

	/* X���ǂ̐�2�񃋁[�v -------------------------------------------------- */
	for (int iIndex = 0; iIndex < 31; iIndex += 30) {
		YWall(&iBox[0][iIndex], 31);/* Y���ǐ��� */
	}

	/* Y���ǂ̐�2�񃋁[�v -------------------------------------------------- */
	for (int iIndex = 0; iIndex < 31; iIndex += 30) {
		XWall(&iBox[iIndex][0], 31);/* X���ǐ��� */
	}
	ptr = START(&iBox[0][0]);/* �J�n�_���� */
	Movement(ptr, &iBox[0][0]);/* ���H���� */
	iArrayPrint(&iBox[0][0], 31 * 31);/* int�^�z��\�� */
	Next();
	system("cls");
	printf("\x1b[93m�퓬�J�n\x1b[39m\n\n\n");
	Next();
	system("cls");

	/* 1P�A2P�ǂ��炩��HP��0�ȉ��ɂȂ�܂Ń��[�v --------------------------- */
	while ((Info[0].HP > 0) && (Info[1].HP > 0)) {
		iCount++;
		printf("TURN\x1b[31m%d\x1b[39m\n", iCount);
		iArrayReset(&iCritSw[0], 2);/* ��S�̈ꌂ����z�񏉊��� */

		/* �\���̔z��̐����[�v -------------------------------------------- */
		for (int iIndex1 = 0, iIndex2 = 1; iIndex1 < 2; iIndex1++, iIndex2--) {
			iDamage[iIndex1] = Damage(&Info[iIndex1], &Info[iIndex2], &iCritSw[iIndex1]);/* �^�_���[�W�Z�o */

			/* ��U(2P)��HP��0�ȉ��Ȃ�ΐ퓬�s�\��Ԃɂ��� */
			if (Info[1].HP <= 0) {
				iDeadSw++;
				break;
			}
		}

		/* �\���̔z��̐����[�v -------------------------------------------- */
		for (int iIndex1 = 0, iIndex2 = 1; iIndex1 < 2; iIndex1++, iIndex2--) {
			if (iDeadSw == 0) {
				CPrint(iCritSw[iIndex2]);/* ��S�̈ꌂ�\�� */
				DPrint(&Info[iIndex1], &Info[iIndex2], iDamage[iIndex2], iIndex1, iIndex2);/* �^�_���[�W�\�� */
			}
			else {
				iDeadSw--;
			}
			SPrint(&Info[iIndex1], iIndex1, iMaxHP[iIndex1]);/* �X�e�[�^�X���\�� */
		}
		Next();
		system("cls");
	}
	
	RPrint(&Info[0], &Info[1]);/* �퓬���ʕ\�� */

	/* �����I�� */
	rewind(stdin);
	getchar();
	return 0;
}

/* �X�e�[�^�X�U�蕪���֐� */
int Status(struct PLAYER* fInfo, int fCount) {
	int fMaxHP = 0;								/* �ő�HP�L�^(�߂�l)		 */
	int fWGacha = 0;							/* ����K�`��				 */
	int flength = 0;							/* ������					 */
	char fConfirm = 0;

	/* �������[�v ---------------------------------------------------------- */
	while(1) {
		rewind(stdin);
		fgets(fInfo[0].NAME, sizeof(fInfo[42].NAME), stdin);

		/* 10Byte�ڂ�0(null)�܂��͖����͂Ȃ�΍ē��͂𑣂�(SHIFT_JIS) */
		if ((fInfo->NAME[10] == 0) && (fInfo->NAME[0] != '\n')) {
			printf("\x1b[31m(�I)\x1b[39m�v���C���[����\x1b[31m10Byte�ȏ�\x1b[39m���͂��Ă��������B(���p1����1Byte�A�S�p1����2Byte)\n\n�ē���:");
		}
		else if (fInfo->NAME[0] == '\n') {
			printf("\x1b[31m(�I)������\x1b[39m�ł��B\n\n�ē���:");
		}

		/* 41Byte�ڂ��Ƀf�[�^�������Ă���Ȃ�ē��͂𑣂�(SHIFT_JIS) */
		if (fInfo->NAME[41] != 0) {
			printf("\x1b[31m(�I)\x1b[39m�v���C���[����\x1b[31m40Byte�ȓ���\x1b[39m���͂��Ă��������B(���p1����1Byte�A�S�p1����2Byte)\n\n�ē���:");
			cArrayReset(&fInfo->NAME[0], 43);/* ���O�z�񏉊��� */
		}

		/* ��L�����̓��͏����𖞂������ꍇ�A�m�F��ʂ�\�� */
		if ((fInfo->NAME[10] != 0) && (fInfo->NAME[41] == 0)) {
			printf("\n�ȍ~�ύX���ł��܂���B�{���ɓ��͂��m�肵�܂���?\n[Y]�͂� [N]������\n\n");

			/* Y��N�����͂����܂Ń��[�v ---------------------------------- */
			do {
				scanf("%c", &fConfirm);
				rewind(stdin);

				/* ���͂�Y��N����Ȃ���΍ē��͂𑣂� */
				if ((fConfirm != 'Y') && (fConfirm != 'N')) {
					printf("\x1b[31m(�I)\x1b[39m\x1b[31m[Y]\x1b[39m�͂� \x1b[31m[N]\x1b[39m�������œ��͂��Ă��������B\n\n");
				}
			} while ((fConfirm != 'Y') && (fConfirm != 'N'));

			/* ���͂�Y�Ȃ�Ζ������[�v�𔲂��AN�Ȃ�΍ē��͂𑣂� */
			if (fConfirm == 'Y') {
				break;
			}
			else if (fConfirm == 'N') {
				printf("\n�ē���:");
				cArrayReset(&fInfo->NAME[0], 43);/* ���O�z�񏉊��� */
			}
		}
	}
	flength = strlen(fInfo->NAME);

	/* ���͕����̏I�[-1�Ԗڂ̔z�񂪉��s�Ȃ��0�ɂ��� */
	if (fInfo->NAME[flength - 1] == '\n') {
		fInfo->NAME[flength - 1] = 0;
	}

	fInfo->HP = (unsigned char)fInfo->NAME[2] * (rand() % 4 + 1) % 999 + 1;

	/* TURN0(�X�e�[�^�X�U�蕪����)��HP���ő�HP�Ƃ��ĕۑ� */
	if (fCount == 0) {
		fMaxHP = fInfo->HP;
	}
	fInfo->ATK = ((unsigned char)fInfo->NAME[3] + (unsigned char)fInfo->NAME[4] * (rand() % 4 + 1)) % 255 + 1;
	fInfo->DEF = ((unsigned char)fInfo->NAME[5] + (unsigned char)fInfo->NAME[4] * (rand() % 2 + 1)) % 127 + 1;
	fInfo->LUCK = (unsigned char)fInfo->NAME[8] * (rand() % 16 + 5) % 999 + 1;
	fWGacha = rand() % 8;

	/* ����K�`���̒l�ɉ��������햼�������o����z��Ɋi�[ */
	if (fWGacha == 0) {
		strcpy(fInfo->WEAPON, "�S���t�{�[��");
	}
	if (fWGacha == 1) {
		strcpy(fInfo->WEAPON, "�V�i");
	}
	if (fWGacha == 2) {
		strcpy(fInfo->WEAPON, "���K�l");
	}
	if (fWGacha == 3) {
		strcpy(fInfo->WEAPON, "�A�b�N�X");
	}
	if (fWGacha == 4) {
		strcpy(fInfo->WEAPON, "���Õi");
	}
	if (fWGacha == 5) {
		strcpy(fInfo->WEAPON, "���[�\�N");
	}
	if (fWGacha == 6) {
		strcpy(fInfo->WEAPON, "�J����");
	}
	if (fWGacha == 7) {
		strcpy(fInfo->WEAPON, "�����^�C��");
	}
	return fMaxHP;
}

/* �X�e�[�^�X���\���֐� */
void SPrint(struct PLAYER* fInfo, int fIndex, int fMaxHP) {
	int fPColor = 0;							/* �v���C���[�\���F����		 */
	int fHPColor = 0;							/* HP�\���F����				 */
	fPColor = PColorPrint(fIndex);/* �v���C���[�\���F�t�� */
	fHPColor = HPColorPrint(fInfo, fMaxHP);/* HP�\���F�t�� */
	printf("\x1b[%dm%dP:%s\x1b[39m\n", fPColor, fIndex + 1, &fInfo->NAME[0]);
	printf("HP:\x1b[%dm%d/%d\x1b[39m\n", fHPColor, fInfo->HP, fMaxHP);
	printf("ATK:%d\n", fInfo->ATK);
	printf("DEF:%d\n", fInfo->DEF);
	printf("LUCK:%d\n", fInfo->LUCK);
	printf("WEAPON:%s\n\n\n", &fInfo->WEAPON[0]);
}

/* �^�_���[�W�Z�o�֐� */
int Damage(struct PLAYER* fInfo1, struct PLAYER* fInfo2, int* ptr) {
	int fDamage = 0;							/* �^�_���[�W�v�Z(�߂�l)	 */
	int fCritRate = 0;							/* ��S���v�Z				 */
	int fRate = 0;								/* �m���v�Z					 */
	fDamage = fInfo1->ATK - rand() % fInfo2->DEF + 1;

	/* �_���[�W��0�ȉ��̏ꍇ�A0~5�̒l�������_���ŏo�� */
	if (fDamage <= 0) {
		fDamage = rand() % 6;
	}
	fCritRate = (fInfo1->LUCK) / 50;
	fCritRate = (fCritRate + 1) * 2;
	fRate = 100 / fCritRate;

	/* �]�肪����Ίm���v�Z���ʂ�+1*/
	if (100 % fRate >= 1) {
		fRate++;
	}
	fRate = (rand() % fRate + 1) * fCritRate;

	/* �m���v�Z���ʂ�100�ȏ�̏ꍇ�A��S�̈ꌂ���� */
	if (fRate >= 100) {
		fDamage = fDamage * (rand() % 24 + 12) / 10;
		*ptr = 1;
	}
	fInfo2->HP -= fDamage;
	return fDamage;
}

/* ��S�̈ꌂ�\���֐� */
void CPrint(int fCritSw) {

	/* ��S�̈ꌂ���䂪1�̂Ƃ��\�������ɓ��� */
	if (fCritSw == 1) {
		printf("\x1b[31m��S�̈ꌂ�I\x1b[39m");
	}
}

/* �^�_���[�W�\���֐� */
void DPrint(struct PLAYER* fInfo1, struct PLAYER* fInfo2, int fDamage, int fIndex1, int fIndex2) {
	int fiColor1 = 0;							/* �\���F����				 */
	int fiColor2 = 0;
	fiColor1 = PColorPrint(fIndex1);
	fiColor2 = PColorPrint(fIndex2);
	printf("\x1b[%dm%s\x1b[39m��", fiColor1, &fInfo1->NAME);
	printf("\x1b[%dm%s\x1b[39m����", fiColor2, &fInfo2->NAME);
	printf("\x1b[31m%d�_���[�W\x1b[39m�󂯂��I\n\n", fDamage);
}

/* �퓬���ʕ\���֐� */
void RPrint(struct PLAYER* fInfo1, struct PLAYER* fInfo2) {

	/* 1P,2P�ǂ��炩��HP��0�ȉ��ɂȂ����Ƃ����s��\������ */
	if (fInfo1->HP <= 0) {
		printf("\x1b[91m%s\x1b[39m�͓|�ꂽ�I\n\n\n", &fInfo1->NAME);
		Next();
		system("cls");
		printf("\x1b[93m�퓬�I��\x1b[39m\n\n\n");
		Next();
		system("cls");
		printf("\x1b[94m%s\x1b[39m�̏����I", &fInfo2->NAME);
	}
	else if (fInfo2->HP <= 0) {
		printf("\x1b[94m%s\x1b[39m�͓|�ꂽ�I\n\n\n", &fInfo2->NAME);
		Next();
		system("cls");
		printf("\x1b[93m�퓬�I��\x1b[39m\n\n\n");
		Next();
		system("cls");
		printf("\x1b[91m%s\x1b[39m�̏����I", &fInfo1->NAME);
	}
}

/* �v���C���[�\���F�t���֐� */
int PColorPrint(int fIndex) {
	int fPColor = 0;							/* �v���C���[�\���F����		 */

	/* �z��0�ԖڂȂ�ԐF(��) */
	if (fIndex == 0) {
		fPColor = 91;
	}

	/* �z��1�ԖڂȂ�F(��) */
	if (fIndex == 1) {
		fPColor = 94;
	}
	return fPColor;
}

/* HP�\���F�t���֐� */
int HPColorPrint(struct PLAYER* fInfo, int fMaxHP) {
	int fHPColor = 0;							/* HP�\���F����		 */

	/* ��HP���ő�HP��50%����Ȃ�ΐF(��) */
	if (fInfo->HP > fMaxHP / 2) {
		fHPColor = 32;
	}

	/* ��HP���ő�HP��50%�ȉ�����20%����Ȃ物�F(��) */
	if ((fInfo->HP <= fMaxHP / 2) && (fInfo->HP > fMaxHP / 5)) {
		fHPColor = 33;
	}

	/* ��HP���ő�HP��20%�ȉ��Ȃ�ԐF(��) */
	if (fInfo->HP <= fMaxHP / 5) {
		fHPColor = 31;
	}
	return fHPColor;
}

/* Y���ǐ����֐� */
void YWall(int* ptr, int fMax) {

	/* Y���̐����[�v ------------------------------------------------------- */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 2;
		ptr += 31;
	}
}

/* X���ǐ����֐� */
void XWall(int* ptr, int fMax) {

	/* X���̐����[�v ------------------------------------------------------- */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 2;
		ptr++;
	}
}

/* �J�n�_�����֐� */
int* START(int* ptr) {
	int fIndex1 = 0;							/* �Y��1(y��)				 */
	int fIndex2 = 0;							/* �Y��2(x��)				 */

	/* �J�n�_�������}�X�ɂȂ�܂Ń��[�v ------------------------------------ */
	do {
		fIndex1 = rand() % 29 + 1;
		fIndex2 = rand() % 29 + 1;
	} while ((fIndex1 % 2 != 1) || (fIndex2 % 2 != 1));
	ptr += fIndex2 + fIndex1 * 31;
	*ptr = 1;
	return ptr;
}

/* ���H�����֐� */
void Movement(int* ptr1, int* ptr3) {
	int* ptr2 = 0;								/* �|�C���^�R�s�[			 */
	int fIndex1 = 0;							/* �Y��1(y��)				 */
	int fIndex2 = 0;							/* �Y��2(x��)				 */
	int fUDLR = 0;								/* �㉺���E�ړ�����			 */
	int fCouseSW = 0;							/* �o�H�؂�ւ�				 */
	int fCount = 0;								/* �������[�v�񐔃J�E���^	 */

	/* �������[�v ---------------------------------------------------------- */
	while (1) {
		ptr2 = ptr1;
		fUDLR = rand() % 4;

		/* UDLR�̒l�ɂ���ď㉺���E�ړ��𐧌� */
		if (fUDLR == 0) {
			ptr1 -= 62;
			ptr2 -= 31;

			/* 2�}�X�悪1�܂���1�}�X�悪2�Ȃ�ړ������A0�Ȃ�ړ����� */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 += 62;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2�񃋁[�v���A2�}�X�i�� ---------------------------------- */
				for (int iCount = 0; iCount < 2; iCount++) {
					*ptr1 = 1;
					ptr1 += 31;
				}
				ptr1 -= 62;
				fCouseSW = 0;
			}
		}
		else if (fUDLR == 1) {
			ptr1 += 62;
			ptr2 += 31;

			/* 2�}�X�悪1�܂���1�}�X�悪2�Ȃ�ړ������A0�Ȃ�ړ����� */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 -= 62;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2�񃋁[�v���A2�}�X�i�� ---------------------------------- */
				for (int iCount = 0; iCount < 2; iCount++) {
					*ptr1 = 1;
					ptr1 -= 31;
				}
				ptr1 += 62;
				fCouseSW = 0;
			}
		}
		else if (fUDLR == 2) {
			ptr1 -= 2;
			ptr2 -= 1;

			/* 2�}�X�悪1�܂���1�}�X�悪2�Ȃ�ړ������A0�Ȃ�ړ����� */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 += 2;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2�񃋁[�v��(2�}�X�i��) ---------------------------------- */
				for (int iCount = 0; iCount < 2; iCount++) {
					*ptr1 = 1;
					ptr1 += 1;
				}
				ptr1 -= 2;
				fCouseSW = 0;
			}
		}
		else if (fUDLR == 3) {
			ptr1 += 2;
			ptr2 += 1;

			/* 2�}�X�悪1�܂���1�}�X�悪2�Ȃ�ړ������A0�Ȃ�ړ����� */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 -= 2;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2�񃋁[�v���A2�}�X�i�� ---------------------------------- */
				for (int iCount = 0; iCount < 2; iCount++) {
					*ptr1 = 1;
					ptr1 -= 1;
				}
				ptr1 += 2;
				fCouseSW = 0;
			}
		}

		/* �o�H�؂�ւ���4�ɂȂ�����ʃ��[�g��T�� */
		if (fCouseSW == 4) {

			/* �|�C���^�̒��g��1����Ȃ��ԃ��[�v --------------------------- */
			do {
				ptr1 = ptr3;

				/* �ʃ��[�g�̊J�n�_�������}�X�ɂȂ�܂Ń��[�v--------------  */
				do {
					fIndex1 = rand() % 29 + 1;
					fIndex2 = rand() % 29 + 1;
				} while ((fIndex1 % 2 != 1) || (fIndex2 % 2 != 1));
				ptr1 += fIndex2 + fIndex1 * 31;
			} while (*ptr1 != 1);
			fCouseSW = 0;
		}
		fCount++;

		/* �������ɂ������H�o���Ƃ���v���񐔂ɂȂ����疳�����[�v�𔲂��� */
		if (fCount > 100000) {
			break;
		}
	}
}

/* int�^�z��\���֐� */
void iArrayPrint(int* ptr, int fMax) {
	for (int fIndex1 = 0, fIndex2 = 0; fIndex1 < fMax; fIndex1++, fIndex2++) {

		/* 31�񃋁[�v1���I�����邲�Ƃɉ��s */
		if (fIndex2 == 31) {
			printf("\n");
			fIndex2 = 0;
		}

		/* �|�C���^�̒��g��0�܂���2�Ȃ�΁����o�͂��A1�Ȃ�΁����o�͂��� */
		if ((*ptr == 0) || (*ptr == 2)) {
			printf("��");
		}
		else if (*ptr == 1) {
			printf("��");
		}
		ptr++;
	}
}

/* int�^�z�񏉊����֐� */
void iArrayReset(int* ptr, int fMax) {

	/* �z��̐����[�v ------------------------------------------------------ */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 0;
		ptr++;
	}
}

/* char�^�z�񏉊����֐� */
void cArrayReset(char* ptr, int fMax) {

	/* �z��̐����[�v ------------------------------------------------------ */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 0;
		ptr++;
	}
}

/* ���̉�ʕ\���֐� */
void Next() {
	char fInput = 0;							/* ����						 */
	printf("[ENTER]�Ŏ��̉�ʂɐi�ށB");

	/* ���s����͂���܂Ń��[�v -------------------------------------------- */
	while (fInput != '\n') {
		rewind(stdin);
		scanf("%c", &fInput);
	}
}
/* ------------------------------------------------------------------------- */
/*	Copyright International Professional University of Technology in Osaka	 */
/* ------------------------------------------------------------------------- */