/**
* @file Game.cpp
* @brief 課題9 : Gameクラスメンバー関数
* @author IT11H308 ok230112 江藤大晴
* @date 2023/12/1
*/

#include <iostream>
#include <string>

#include "Game.h"
#include "Enemy.h"


using namespace std;

/* コンストラクタ */
Game::Game() {

	/* 敵オブジェクトを管理用ポインタ配列をNULLで初期化 */
	for (int index = 0; index < ENEMY_NUM_MAX; index++) {
		enemy[index] = NULL;
	}
	
	/* 敵数を0で初期化 */
	enemy_num = 0;
}

/* 実行 */
void Game::execute() {

	/* 初期化 */
	initialize();

	/* 戦闘 */
	battle();

	/* 終了 */
	finalize();
}

/* 初期化 */
void Game::initialize() {

	/* 敵数を乱数で決める */
	enemy_num = ENEMY_NUM_MIN + rnd() % (ENEMY_NUM_MAX - ENEMY_NUM_MIN + 1);

	/* 敵数分繰り返す */
	for (int index = 0; index < enemy_num; index++) {

		/* 敵オブジェクトを生成 */
		enemy[index] = new Enemy;

		/* 敵オブジェクトの名前を設定 */
		string enemy_name = "スライム" + to_string(index);
		enemy[index]->setName(enemy_name);

		/* 敵オブジェクトのHPを設定 */
		int enemy_hp = ENEMY_HP_MIN + rnd() % (ENEMY_HP_MAX - ENEMY_HP_MIN + 1);
		enemy[index]->setHitPoint(enemy_hp);
	}
}

/* 戦闘 */
void Game::battle() {

	/* 戦闘終了まで繰り返す */
	while (1) {

		/* 全ての敵の状態（名前、HP）を表示 */
		for (int index = 0; index < enemy_num; index++) {
			if (enemy[index]->getHitPoint() > 0) {
				cout << index << ":";
				enemy[index]->showData();
			}
		}

		/* プレイヤーのコマンドを入力 */
		cout << "どのスライムを攻撃する？（数字で入力）999=終了" << endl;
		int cmd;
		cin >> cmd;

		/* 入力値が終了なら戦闘終了 */
		if (cmd == 999) {
			return;
		}
		
		/* 入力値が有効か判定 */
		if ((cmd >= enemy_num) || (cmd < enemy_num - enemy_num) || (enemy[cmd]->getHitPoint() == 0)) {
			cout << "無効な値です" << endl;
		}
		else {

			/* 指定した敵にダメージを与える */
			int damage = rnd() % ATTACK_MAX;
			enemy[cmd]->receiveDamage(damage);

			/* 全ての敵が倒れたら終了 */
			/* 全敵のHP合計が0なら全滅と判定 */
			int sum = 0;
			for (int index = 0; index < enemy_num; index++) {
				sum += enemy[index]->getHitPoint();
			}
			if (sum == 0) {
				cout << "全てのスライムを倒した！" << endl;
				return;
			}
		}
	}
}

/* 終了 */
void Game::finalize() {

	/* 敵オブジェクトを破棄 */
	for (int index = 0; index < enemy_num; index++) {
		delete enemy[index];
	}
}