/* ------------------------------------------------------------------------- */
/* 課題No,10_文字列バトラー_江藤大晴.cpp									 */
/* AssignmentTenBattler														 */
/* 入力文字列によってステータスが決まるターン制バトル						 */
/*																			 */
/* ------------------------------------------------------------------------- */
/*	番号	更新履歴								日付		氏名		 */
/* ------------------------------------------------------------------------- */
/*	000000	新規作成								2023/08/10	江藤  大晴	 */
/* ------------------------------------------------------------------------- */
#define _CRT_SECURE_NO_WARNINGS					/* scanf対応				 */

/* ------------------------------------------------------------------------- */
/* includeファイル															 */
/* ------------------------------------------------------------------------- */
#include<stdio.h>								/* 標準入出力制御			 */
#include<time.h>								/* 時間制御					 */
#include<stdlib.h>								/* 基本ライブラリ			 */
#include <string.h>								/* 文字列制御				 */

/* ------------------------------------------------------------------------- */
/* 構造体定義																 */
/* ------------------------------------------------------------------------- */
struct PLAYER {
	int HP;										/* 体力						 */
	int ATK;									/* 攻撃力					 */
	int DEF;									/* 運						 */
	int LUCK;									/* 防御力					 */
	char NAME[43];								/* プレイヤー名				 */
	char WEAPON[41];							/* 武器						 */
};

/* ------------------------------------------------------------------------- */
/* プロトタイプ宣言															 */
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
/* 関数名	: main															 */
/* 機能名	: ターン制バトル												 */
/* 機能概要 : 入力文字列によってステータスが決まるターン制バトル			 */
/* 引数		: void	: f無し	:												 */
/* 戻り値	: int	: OSへ返す値(今は0固定)									 */
/* 作成日	: 2023/08/10		江藤  大晴		新規作成					 */
/* ------------------------------------------------------------------------- */
int main(void)
{
	/* 変数宣言 ------------------------------------------------------------ */
	struct PLAYER Info[2] = {					/* 構造体配列				 */
	{/* PLAYER配列0番目 ---------------------------------------------------- */
		0,										/* HP						 */
		0,										/* ATK						 */
		0,										/* DEF						 */
		0										/* LUCK						 */
	},
	{/* PLAYER配列1番目 ---------------------------------------------------- */
		0,										/* HP						 */
		0,										/* ATK						 */
		0,										/* DEF						 */
		0										/* LUCK						 */
	}
	};
	int iMaxHP[2]{ 0,0 };						/* 最大HP記録配列			 */
	int iDamage[2]{ 0,0 };						/* 与ダメージ計算配列		 */
	int iCritSw[2]{ 0,0 };						/* 会心の一撃制御配列		 */
	int iDeadSw = 0;							/* 戦闘不能制御(2P)			 */
	int iCount = 0;								/* ターン数カウンタ			 */
	int iPColor = 0;							/* プレイヤー表示色制御		 */
	int iBox[31][31];							/* データ2次元配列(31*31)	 */
	int* ptr = 0;								/* ポインタ					 */
	srand(time(NULL));							/* 乱数初期化				 */

	/* 処理開始------------------------------------------------------------- */

	/* 構造体配列の数ループ ------------------------------------------------ */
	for (int iIndex = 0; iIndex < 2; iIndex++) {
		cArrayReset(&Info[iIndex].NAME[0], 43);/* 名前配列初期化 */
		cArrayReset(&Info[iIndex].WEAPON[0], 41);/* 武器配列初期化 */
	}
	
	printf("対応文字コード:SHIFT_JIS(半角1文字1Byte、全角1文字2Byte)\n");

	/* 構造体配列の数ループ ------------------------------------------------ */
	for (int iIndex = 0; iIndex < 2; iIndex++) {
		iPColor = PColorPrint(iIndex);/* プレイヤー表示色付け */
		printf("\nプレイヤー名を入力してください。(10Byte以上、40byte以内)\n[ENTER]で入力を確定する。\n\n\x1b[%dm%dP:\x1b[39m",iPColor, iIndex + 1);
		iMaxHP[iIndex] = Status(&Info[iIndex], iCount);/* ステータス振り分け */
	}
	system("cls");
	printf("ステータス情報\n\n\n");

	/* 構造体配列の数ループ ------------------------------------------------ */
	for (int iIndex = 0; iIndex < 2; iIndex++) {
		SPrint(&Info[iIndex], iIndex, iMaxHP[iIndex]);/* ステータス情報表示 */
	}
	Next();
	system("cls");
	iArrayReset(&iBox[0][0], 31 * 31);/* int型配列初期化 */

	/* X軸壁の数2回ループ -------------------------------------------------- */
	for (int iIndex = 0; iIndex < 31; iIndex += 30) {
		YWall(&iBox[0][iIndex], 31);/* Y軸壁生成 */
	}

	/* Y軸壁の数2回ループ -------------------------------------------------- */
	for (int iIndex = 0; iIndex < 31; iIndex += 30) {
		XWall(&iBox[iIndex][0], 31);/* X軸壁生成 */
	}
	ptr = START(&iBox[0][0]);/* 開始点決定 */
	Movement(ptr, &iBox[0][0]);/* 迷路生成 */
	iArrayPrint(&iBox[0][0], 31 * 31);/* int型配列表示 */
	Next();
	system("cls");
	printf("\x1b[93m戦闘開始\x1b[39m\n\n\n");
	Next();
	system("cls");

	/* 1P、2PどちらかのHPが0以下になるまでループ --------------------------- */
	while ((Info[0].HP > 0) && (Info[1].HP > 0)) {
		iCount++;
		printf("TURN\x1b[31m%d\x1b[39m\n", iCount);
		iArrayReset(&iCritSw[0], 2);/* 会心の一撃制御配列初期化 */

		/* 構造体配列の数ループ -------------------------------------------- */
		for (int iIndex1 = 0, iIndex2 = 1; iIndex1 < 2; iIndex1++, iIndex2--) {
			iDamage[iIndex1] = Damage(&Info[iIndex1], &Info[iIndex2], &iCritSw[iIndex1]);/* 与ダメージ算出 */

			/* 後攻(2P)のHPが0以下ならば戦闘不能状態にする */
			if (Info[1].HP <= 0) {
				iDeadSw++;
				break;
			}
		}

		/* 構造体配列の数ループ -------------------------------------------- */
		for (int iIndex1 = 0, iIndex2 = 1; iIndex1 < 2; iIndex1++, iIndex2--) {
			if (iDeadSw == 0) {
				CPrint(iCritSw[iIndex2]);/* 会心の一撃表示 */
				DPrint(&Info[iIndex1], &Info[iIndex2], iDamage[iIndex2], iIndex1, iIndex2);/* 与ダメージ表示 */
			}
			else {
				iDeadSw--;
			}
			SPrint(&Info[iIndex1], iIndex1, iMaxHP[iIndex1]);/* ステータス情報表示 */
		}
		Next();
		system("cls");
	}
	
	RPrint(&Info[0], &Info[1]);/* 戦闘結果表示 */

	/* 処理終了 */
	rewind(stdin);
	getchar();
	return 0;
}

/* ステータス振り分け関数 */
int Status(struct PLAYER* fInfo, int fCount) {
	int fMaxHP = 0;								/* 最大HP記録(戻り値)		 */
	int fWGacha = 0;							/* 武器ガチャ				 */
	int flength = 0;							/* 文字数					 */
	char fConfirm = 0;

	/* 無限ループ ---------------------------------------------------------- */
	while(1) {
		rewind(stdin);
		fgets(fInfo[0].NAME, sizeof(fInfo[42].NAME), stdin);

		/* 10Byte目が0(null)または未入力ならば再入力を促す(SHIFT_JIS) */
		if ((fInfo->NAME[10] == 0) && (fInfo->NAME[0] != '\n')) {
			printf("\x1b[31m(！)\x1b[39mプレイヤー名は\x1b[31m10Byte以上\x1b[39m入力してください。(半角1文字1Byte、全角1文字2Byte)\n\n再入力:");
		}
		else if (fInfo->NAME[0] == '\n') {
			printf("\x1b[31m(！)未入力\x1b[39mです。\n\n再入力:");
		}

		/* 41Byte目がにデータが入っているなら再入力を促す(SHIFT_JIS) */
		if (fInfo->NAME[41] != 0) {
			printf("\x1b[31m(！)\x1b[39mプレイヤー名は\x1b[31m40Byte以内で\x1b[39m入力してください。(半角1文字1Byte、全角1文字2Byte)\n\n再入力:");
			cArrayReset(&fInfo->NAME[0], 43);/* 名前配列初期化 */
		}

		/* 上記処理の入力条件を満たした場合、確認画面を表示 */
		if ((fInfo->NAME[10] != 0) && (fInfo->NAME[41] == 0)) {
			printf("\n以降変更ができません。本当に入力を確定しますか?\n[Y]はい [N]いいえ\n\n");

			/* YかNが入力されるまでループ ---------------------------------- */
			do {
				scanf("%c", &fConfirm);
				rewind(stdin);

				/* 入力がYかNじゃなければ再入力を促す */
				if ((fConfirm != 'Y') && (fConfirm != 'N')) {
					printf("\x1b[31m(！)\x1b[39m\x1b[31m[Y]\x1b[39mはい \x1b[31m[N]\x1b[39mいいえで入力してください。\n\n");
				}
			} while ((fConfirm != 'Y') && (fConfirm != 'N'));

			/* 入力がYならば無限ループを抜け、Nならば再入力を促す */
			if (fConfirm == 'Y') {
				break;
			}
			else if (fConfirm == 'N') {
				printf("\n再入力:");
				cArrayReset(&fInfo->NAME[0], 43);/* 名前配列初期化 */
			}
		}
	}
	flength = strlen(fInfo->NAME);

	/* 入力文字の終端-1番目の配列が改行ならば0にする */
	if (fInfo->NAME[flength - 1] == '\n') {
		fInfo->NAME[flength - 1] = 0;
	}

	fInfo->HP = (unsigned char)fInfo->NAME[2] * (rand() % 4 + 1) % 999 + 1;

	/* TURN0(ステータス振り分け時)のHPを最大HPとして保存 */
	if (fCount == 0) {
		fMaxHP = fInfo->HP;
	}
	fInfo->ATK = ((unsigned char)fInfo->NAME[3] + (unsigned char)fInfo->NAME[4] * (rand() % 4 + 1)) % 255 + 1;
	fInfo->DEF = ((unsigned char)fInfo->NAME[5] + (unsigned char)fInfo->NAME[4] * (rand() % 2 + 1)) % 127 + 1;
	fInfo->LUCK = (unsigned char)fInfo->NAME[8] * (rand() % 16 + 5) % 999 + 1;
	fWGacha = rand() % 8;

	/* 武器ガチャの値に応じた武器名をメンバ武器配列に格納 */
	if (fWGacha == 0) {
		strcpy(fInfo->WEAPON, "ゴルフボール");
	}
	if (fWGacha == 1) {
		strcpy(fInfo->WEAPON, "新品");
	}
	if (fWGacha == 2) {
		strcpy(fInfo->WEAPON, "メガネ");
	}
	if (fWGacha == 3) {
		strcpy(fInfo->WEAPON, "アックス");
	}
	if (fWGacha == 4) {
		strcpy(fInfo->WEAPON, "中古品");
	}
	if (fWGacha == 5) {
		strcpy(fInfo->WEAPON, "ローソク");
	}
	if (fWGacha == 6) {
		strcpy(fInfo->WEAPON, "カメラ");
	}
	if (fWGacha == 7) {
		strcpy(fInfo->WEAPON, "高級タイヤ");
	}
	return fMaxHP;
}

/* ステータス情報表示関数 */
void SPrint(struct PLAYER* fInfo, int fIndex, int fMaxHP) {
	int fPColor = 0;							/* プレイヤー表示色制御		 */
	int fHPColor = 0;							/* HP表示色制御				 */
	fPColor = PColorPrint(fIndex);/* プレイヤー表示色付け */
	fHPColor = HPColorPrint(fInfo, fMaxHP);/* HP表示色付け */
	printf("\x1b[%dm%dP:%s\x1b[39m\n", fPColor, fIndex + 1, &fInfo->NAME[0]);
	printf("HP:\x1b[%dm%d/%d\x1b[39m\n", fHPColor, fInfo->HP, fMaxHP);
	printf("ATK:%d\n", fInfo->ATK);
	printf("DEF:%d\n", fInfo->DEF);
	printf("LUCK:%d\n", fInfo->LUCK);
	printf("WEAPON:%s\n\n\n", &fInfo->WEAPON[0]);
}

/* 与ダメージ算出関数 */
int Damage(struct PLAYER* fInfo1, struct PLAYER* fInfo2, int* ptr) {
	int fDamage = 0;							/* 与ダメージ計算(戻り値)	 */
	int fCritRate = 0;							/* 会心率計算				 */
	int fRate = 0;								/* 確率計算					 */
	fDamage = fInfo1->ATK - rand() % fInfo2->DEF + 1;

	/* ダメージが0以下の場合、0~5の値をランダムで出す */
	if (fDamage <= 0) {
		fDamage = rand() % 6;
	}
	fCritRate = (fInfo1->LUCK) / 50;
	fCritRate = (fCritRate + 1) * 2;
	fRate = 100 / fCritRate;

	/* 余りがあれば確率計算結果に+1*/
	if (100 % fRate >= 1) {
		fRate++;
	}
	fRate = (rand() % fRate + 1) * fCritRate;

	/* 確率計算結果が100以上の場合、会心の一撃発動 */
	if (fRate >= 100) {
		fDamage = fDamage * (rand() % 24 + 12) / 10;
		*ptr = 1;
	}
	fInfo2->HP -= fDamage;
	return fDamage;
}

/* 会心の一撃表示関数 */
void CPrint(int fCritSw) {

	/* 会心の一撃制御が1のとき表示処理に入る */
	if (fCritSw == 1) {
		printf("\x1b[31m会心の一撃！\x1b[39m");
	}
}

/* 与ダメージ表示関数 */
void DPrint(struct PLAYER* fInfo1, struct PLAYER* fInfo2, int fDamage, int fIndex1, int fIndex2) {
	int fiColor1 = 0;							/* 表示色制御				 */
	int fiColor2 = 0;
	fiColor1 = PColorPrint(fIndex1);
	fiColor2 = PColorPrint(fIndex2);
	printf("\x1b[%dm%s\x1b[39mは", fiColor1, &fInfo1->NAME);
	printf("\x1b[%dm%s\x1b[39mから", fiColor2, &fInfo2->NAME);
	printf("\x1b[31m%dダメージ\x1b[39m受けた！\n\n", fDamage);
}

/* 戦闘結果表示関数 */
void RPrint(struct PLAYER* fInfo1, struct PLAYER* fInfo2) {

	/* 1P,2PどちらかのHPが0以下になったとき勝敗を表示する */
	if (fInfo1->HP <= 0) {
		printf("\x1b[91m%s\x1b[39mは倒れた！\n\n\n", &fInfo1->NAME);
		Next();
		system("cls");
		printf("\x1b[93m戦闘終了\x1b[39m\n\n\n");
		Next();
		system("cls");
		printf("\x1b[94m%s\x1b[39mの勝利！", &fInfo2->NAME);
	}
	else if (fInfo2->HP <= 0) {
		printf("\x1b[94m%s\x1b[39mは倒れた！\n\n\n", &fInfo2->NAME);
		Next();
		system("cls");
		printf("\x1b[93m戦闘終了\x1b[39m\n\n\n");
		Next();
		system("cls");
		printf("\x1b[91m%s\x1b[39mの勝利！", &fInfo1->NAME);
	}
}

/* プレイヤー表示色付け関数 */
int PColorPrint(int fIndex) {
	int fPColor = 0;							/* プレイヤー表示色制御		 */

	/* 配列0番目なら赤色(明) */
	if (fIndex == 0) {
		fPColor = 91;
	}

	/* 配列1番目なら青色(明) */
	if (fIndex == 1) {
		fPColor = 94;
	}
	return fPColor;
}

/* HP表示色付け関数 */
int HPColorPrint(struct PLAYER* fInfo, int fMaxHP) {
	int fHPColor = 0;							/* HP表示色制御		 */

	/* 現HPが最大HPの50%より上なら緑色(暗) */
	if (fInfo->HP > fMaxHP / 2) {
		fHPColor = 32;
	}

	/* 現HPが最大HPの50%以下かつ20%より上なら黄色(暗) */
	if ((fInfo->HP <= fMaxHP / 2) && (fInfo->HP > fMaxHP / 5)) {
		fHPColor = 33;
	}

	/* 現HPが最大HPの20%以下なら赤色(暗) */
	if (fInfo->HP <= fMaxHP / 5) {
		fHPColor = 31;
	}
	return fHPColor;
}

/* Y軸壁生成関数 */
void YWall(int* ptr, int fMax) {

	/* Y軸の数ループ ------------------------------------------------------- */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 2;
		ptr += 31;
	}
}

/* X軸壁生成関数 */
void XWall(int* ptr, int fMax) {

	/* X軸の数ループ ------------------------------------------------------- */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 2;
		ptr++;
	}
}

/* 開始点生成関数 */
int* START(int* ptr) {
	int fIndex1 = 0;							/* 添字1(y軸)				 */
	int fIndex2 = 0;							/* 添字2(x軸)				 */

	/* 開始点が偶数マスになるまでループ ------------------------------------ */
	do {
		fIndex1 = rand() % 29 + 1;
		fIndex2 = rand() % 29 + 1;
	} while ((fIndex1 % 2 != 1) || (fIndex2 % 2 != 1));
	ptr += fIndex2 + fIndex1 * 31;
	*ptr = 1;
	return ptr;
}

/* 迷路生成関数 */
void Movement(int* ptr1, int* ptr3) {
	int* ptr2 = 0;								/* ポインタコピー			 */
	int fIndex1 = 0;							/* 添字1(y軸)				 */
	int fIndex2 = 0;							/* 添字2(x軸)				 */
	int fUDLR = 0;								/* 上下左右移動制御			 */
	int fCouseSW = 0;							/* 経路切り替え				 */
	int fCount = 0;								/* 無限ループ回数カウンタ	 */

	/* 無限ループ ---------------------------------------------------------- */
	while (1) {
		ptr2 = ptr1;
		fUDLR = rand() % 4;

		/* UDLRの値によって上下左右移動を制御 */
		if (fUDLR == 0) {
			ptr1 -= 62;
			ptr2 -= 31;

			/* 2マス先が1または1マス先が2なら移動せず、0なら移動する */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 += 62;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2回ループし、2マス進む ---------------------------------- */
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

			/* 2マス先が1または1マス先が2なら移動せず、0なら移動する */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 -= 62;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2回ループし、2マス進む ---------------------------------- */
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

			/* 2マス先が1または1マス先が2なら移動せず、0なら移動する */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 += 2;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2回ループし(2マス進む) ---------------------------------- */
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

			/* 2マス先が1または1マス先が2なら移動せず、0なら移動する */
			if ((*ptr1 == 1) || (*ptr2 == 2)) {
				ptr1 -= 2;
				fCouseSW++;
			}
			else if (*ptr1 == 0) {

				/* 2回ループし、2マス進む ---------------------------------- */
				for (int iCount = 0; iCount < 2; iCount++) {
					*ptr1 = 1;
					ptr1 -= 1;
				}
				ptr1 += 2;
				fCouseSW = 0;
			}
		}

		/* 経路切り替えが4になったら別ルートを探す */
		if (fCouseSW == 4) {

			/* ポインタの中身が1じゃない間ループ --------------------------- */
			do {
				ptr1 = ptr3;

				/* 別ルートの開始点が偶数マスになるまでループ--------------  */
				do {
					fIndex1 = rand() % 29 + 1;
					fIndex2 = rand() % 29 + 1;
				} while ((fIndex1 % 2 != 1) || (fIndex2 % 2 != 1));
				ptr1 += fIndex2 + fIndex1 * 31;
			} while (*ptr1 != 1);
			fCouseSW = 0;
		}
		fCount++;

		/* さすがにもう迷路出来とるやろ思う回数になったら無限ループを抜ける */
		if (fCount > 100000) {
			break;
		}
	}
}

/* int型配列表示関数 */
void iArrayPrint(int* ptr, int fMax) {
	for (int fIndex1 = 0, fIndex2 = 0; fIndex1 < fMax; fIndex1++, fIndex2++) {

		/* 31回ループ1が終了するごとに改行 */
		if (fIndex2 == 31) {
			printf("\n");
			fIndex2 = 0;
		}

		/* ポインタの中身が0または2ならば■を出力し、1ならば□を出力する */
		if ((*ptr == 0) || (*ptr == 2)) {
			printf("■");
		}
		else if (*ptr == 1) {
			printf("□");
		}
		ptr++;
	}
}

/* int型配列初期化関数 */
void iArrayReset(int* ptr, int fMax) {

	/* 配列の数ループ ------------------------------------------------------ */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 0;
		ptr++;
	}
}

/* char型配列初期化関数 */
void cArrayReset(char* ptr, int fMax) {

	/* 配列の数ループ ------------------------------------------------------ */
	for (int fIndex = 0; fIndex < fMax; fIndex++) {
		*ptr = 0;
		ptr++;
	}
}

/* 次の画面表示関数 */
void Next() {
	char fInput = 0;							/* 入力						 */
	printf("[ENTER]で次の画面に進む。");

	/* 改行を入力するまでループ -------------------------------------------- */
	while (fInput != '\n') {
		rewind(stdin);
		scanf("%c", &fInput);
	}
}
/* ------------------------------------------------------------------------- */
/*	Copyright International Professional University of Technology in Osaka	 */
/* ------------------------------------------------------------------------- */