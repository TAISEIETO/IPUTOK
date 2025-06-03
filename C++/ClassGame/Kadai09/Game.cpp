/**
* @file Game.cpp
* @brief �ۑ�9 : Game�N���X�����o�[�֐�
* @author IT11H308 ok230112 �]���吰
* @date 2023/12/1
*/

#include <iostream>
#include <string>

#include "Game.h"
#include "Enemy.h"


using namespace std;

/* �R���X�g���N�^ */
Game::Game() {

	/* �G�I�u�W�F�N�g���Ǘ��p�|�C���^�z���NULL�ŏ����� */
	for (int index = 0; index < ENEMY_NUM_MAX; index++) {
		enemy[index] = NULL;
	}
	
	/* �G����0�ŏ����� */
	enemy_num = 0;
}

/* ���s */
void Game::execute() {

	/* ������ */
	initialize();

	/* �퓬 */
	battle();

	/* �I�� */
	finalize();
}

/* ������ */
void Game::initialize() {

	/* �G���𗐐��Ō��߂� */
	enemy_num = ENEMY_NUM_MIN + rnd() % (ENEMY_NUM_MAX - ENEMY_NUM_MIN + 1);

	/* �G�����J��Ԃ� */
	for (int index = 0; index < enemy_num; index++) {

		/* �G�I�u�W�F�N�g�𐶐� */
		enemy[index] = new Enemy;

		/* �G�I�u�W�F�N�g�̖��O��ݒ� */
		string enemy_name = "�X���C��" + to_string(index);
		enemy[index]->setName(enemy_name);

		/* �G�I�u�W�F�N�g��HP��ݒ� */
		int enemy_hp = ENEMY_HP_MIN + rnd() % (ENEMY_HP_MAX - ENEMY_HP_MIN + 1);
		enemy[index]->setHitPoint(enemy_hp);
	}
}

/* �퓬 */
void Game::battle() {

	/* �퓬�I���܂ŌJ��Ԃ� */
	while (1) {

		/* �S�Ă̓G�̏�ԁi���O�AHP�j��\�� */
		for (int index = 0; index < enemy_num; index++) {
			if (enemy[index]->getHitPoint() > 0) {
				cout << index << ":";
				enemy[index]->showData();
			}
		}

		/* �v���C���[�̃R�}���h����� */
		cout << "�ǂ̃X���C�����U������H�i�����œ��́j999=�I��" << endl;
		int cmd;
		cin >> cmd;

		/* ���͒l���I���Ȃ�퓬�I�� */
		if (cmd == 999) {
			return;
		}
		
		/* ���͒l���L�������� */
		if ((cmd >= enemy_num) || (cmd < enemy_num - enemy_num) || (enemy[cmd]->getHitPoint() == 0)) {
			cout << "�����Ȓl�ł�" << endl;
		}
		else {

			/* �w�肵���G�Ƀ_���[�W��^���� */
			int damage = rnd() % ATTACK_MAX;
			enemy[cmd]->receiveDamage(damage);

			/* �S�Ă̓G���|�ꂽ��I�� */
			/* �S�G��HP���v��0�Ȃ�S�łƔ��� */
			int sum = 0;
			for (int index = 0; index < enemy_num; index++) {
				sum += enemy[index]->getHitPoint();
			}
			if (sum == 0) {
				cout << "�S�ẴX���C����|�����I" << endl;
				return;
			}
		}
	}
}

/* �I�� */
void Game::finalize() {

	/* �G�I�u�W�F�N�g��j�� */
	for (int index = 0; index < enemy_num; index++) {
		delete enemy[index];
	}
}