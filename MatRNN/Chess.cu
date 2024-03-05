#include "Chess.cuh"

Position::Position()
{
	i = -1;
	j = -1;
}

Position::Position(int pi, int pj)
{
	i = pi;
	j = pj;
}

Position::Position(const Position& p)
{
	i = p.i;
	j = p.j;
}

void Position::sum(int si, int sj)
{
	i += si;
	j += sj;
}

bool Position::is_correct()
{
	return (i >= 0 && j >= 0 && i < 8 && j < 8);
}

bool Position::equals_pos(const Position& pos)
{
	return pos.i == i && pos.j == j;
}

Move::Move()
{
	mov_t = NORMAL;
}

Move::Move(Position pi, Position pf)
{
	pini = pi;
	pfi = pf;
	mov_t = NORMAL;
}

chess_board::chess_board()
{

}

chess_board::chess_board(std::string fen)
{
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			board[i][j] = 0;

	int i = 0;
	int j = 0;
	int pos = 0;
	while (fen[pos] != ' ')
	{
		if (fen[pos] > '0' && fen[pos] < '9') 
			j += fen[pos] - '0';
		
		else if (fen[pos] == '/') {
			j = 0;
			i++;
		}
		else if (fen[pos] == 'p') {
			board[i][j] = -1;
			j++;
		}
		else if (fen[pos] == 'P') {
			board[i][j] = 1;
			j++;
		}
		else if (fen[pos] == 'r') {
			board[i][j] = -2;
			j++;
		}
		else if (fen[pos] == 'R') {
			board[i][j] = 2;
			j++;
		}
		else if (fen[pos] == 'n') {
			board[i][j] = -3;
			j++;
		}
		else if (fen[pos] == 'N') {
			board[i][j] = 3;
			j++;
		}
		else if (fen[pos] == 'b') {
			board[i][j] = -4;
			j++;
		}
		else if (fen[pos] == 'B') {
			board[i][j] = 4;
			j++;
		}
		else if (fen[pos] == 'q') {
			board[i][j] = -5;
			j++;
		}
		else if (fen[pos] == 'Q') {
			board[i][j] = 5;
			j++;
		}
		else if (fen[pos] == 'k') {
			board[i][j] = -6;
			j++;
		}
		else if (fen[pos] == 'K') {
			board[i][j] = 6;
			j++;
		}
		pos++;
	}
	pos++;

	white_turn = fen[pos] == 'w';
	pos += 2;

	while (fen[pos] != ' ')
	{
		black_can_castle_k = false;
		black_can_castle_q = false;
		white_can_castle_k = false;
		white_can_castle_q = false;

		if (fen[pos] == 'q')
			black_can_castle_q = true;
		else if (fen[pos] == 'Q')
			white_can_castle_q = true;
		else if (fen[pos] == 'k')
			black_can_castle_k = true;
		else if (fen[pos] == 'K') 
			white_can_castle_k = true;

		pos++;
	}
	pos++;

	if (fen[pos] != '-')
		last_move.mov_t = EN_PASSANT;

}

chess_board::chess_board(const chess_board& cb)
{
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			board[i][j] = cb.board[i][j];
	white_can_castle_k = cb.white_can_castle_k;
	black_can_castle_q = cb.black_can_castle_q;
	white_can_castle_k = cb.white_can_castle_k;
	black_can_castle_q = cb.black_can_castle_q;
	fifty_moves_counter = cb.fifty_moves_counter;
	white_turn = cb.white_turn;
	last_move = cb.last_move;
}

void chess_board::move_piece(Move m)
{
	if (board[m.pini.i][m.pini.j] == 0)
		return;
	
	if (board[m.pini.i][m.pini.j] == 6) 
	{
		white_can_castle_q = false;
		white_can_castle_k = false;
	}
	else if (board[m.pini.i][m.pini.j] == -6)
	{
		black_can_castle_q = false;
		black_can_castle_k = false;
	}
	else if (board[m.pini.i][m.pini.j] == 2)
	{
		if(m.pini.j == 0 && m.pini.i == 0)
			white_can_castle_q = false;
		else if(m.pini.j == 7 && m.pini.i == 0)
			white_can_castle_k = false;
	}
	else if (board[m.pini.i][m.pini.j] == -2)
	{
		if (m.pini.j == 0 && m.pini.i == 7)
			black_can_castle_q = false;
		else if(m.pini.j == 7 && m.pini.i == 7)
			black_can_castle_k = false;
	}


	if (board[m.pfi.i][m.pfi.j] != 0 || abs(board[m.pini.i][m.pini.j]) == 1)
		fifty_moves_counter = 50;
	else
		fifty_moves_counter--;

	board[m.pfi.i][m.pfi.j] = board[m.pini.i][m.pini.j];
	board[m.pini.i][m.pini.j] = 0;

	//std::cout << m.mov_t << std::endl;

	if (m.mov_t == EN_PASSANT)
		board[m.pini.i][m.pfi.j] = 0;

	else if (m.mov_t == PROMQ)
		board[m.pfi.i][m.pfi.j] = board[m.pfi.i][m.pfi.j] * 5;

	else if (m.mov_t == PROMN)
		board[m.pfi.i][m.pfi.j] = board[m.pfi.i][m.pfi.j] * 3;

	else if (m.mov_t == PROMR)
		board[m.pfi.i][m.pfi.j] = board[m.pfi.i][m.pfi.j] * 2;

	else if (m.mov_t == PROMB)
		board[m.pfi.i][m.pfi.j] = board[m.pfi.i][m.pfi.j] * 4;

	else if (m.mov_t == CASTLK) {
		board[m.pfi.i][m.pfi.j - 1] = board[m.pfi.i][m.pfi.j + 1];
		board[m.pfi.i][m.pfi.j + 1] = 0;
	}
	else if (m.mov_t == CASTLQ) {
		board[m.pfi.i][m.pfi.j + 1] = board[m.pfi.i][m.pfi.j - 2];
		board[m.pfi.i][m.pfi.j - 2] = 0;
	}

	last_move = m;
	white_turn = !white_turn;

}

void chess_board::print_board()
{
	for (int i = 0; i < 8; ++i)
		std::cout << "._";
	std::cout <<"."<< std::endl;

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			std::cout << "|";
			if (board[i][j] == 0) std::cout << " ";
			else if (board[i][j] == 1) std::cout << "p";
			else if (board[i][j] == 2) std::cout << "r";
			else if (board[i][j] == 3) std::cout << "n";
			else if (board[i][j] == 4) std::cout << "b";
			else if (board[i][j] == 5) std::cout << "q";
			else if (board[i][j] == 6) std::cout << "k";
			else if (board[i][j] == -1) std::cout << "P";
			else if (board[i][j] == -2) std::cout << "R";
			else if (board[i][j] == -3) std::cout << "N";
			else if (board[i][j] == -4) std::cout << "B";
			else if (board[i][j] == -5) std::cout << "Q";
			else if (board[i][j] == -6) std::cout << "K";
			else std::cout << board[i][j];
		}
		std::cout << "|" << std::endl;
	}
	std::cout << std::endl;

}

std::vector<Move> chess_board::generate_semi_legal_moves(bool castle_moves)
{
	std::vector<Move> possible_moves;
	if (fifty_moves_counter <= 0)
		return possible_moves;

	check = false;
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++) {
			Position pos(i, j);
			int piece = board[i][j];

			if (piece == 0) continue;

			if ((white_turn && piece == 5) || (!white_turn && piece == -5))
				generate_queen_moves(pos, possible_moves);
			else if ((white_turn && piece == 3) || (!white_turn && piece == -3))
				generate_knight_moves(pos, possible_moves);
			else if ((white_turn && piece == 2) || (!white_turn && piece == -2))
				generate_rook_moves(pos, possible_moves);
			else if ((white_turn && piece == 4) || (!white_turn && piece == -4))
				generate_bishop_moves(pos, possible_moves);
			else if ((white_turn && piece == 1) || (!white_turn && piece == -1))
				generate_pawn_moves(pos, possible_moves);

		}

	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++) {
			int piece = board[i][j];

			if ((white_turn && piece == 6) || (!white_turn && piece == -6))
				generate_king_moves(Position(i, j), possible_moves, castle_moves);

		}

	return possible_moves;
}

bool chess_board::same_team(Position p1, Position p2)
{
	return ((board[p1.i][p1.j] < 0 && board[p2.i][p2.j] < 0) || (board[p1.i][p1.j] > 0 && board[p2.i][p2.j] > 0));
}

void chess_board::generate_pawn_moves(Position pos, std::vector<Move>& possible_moves)
{
	int piece = board[pos.i][pos.j];

	Position aux_pos(pos);
	aux_pos.sum(-piece, 0);

	if (aux_pos.is_correct() && board[aux_pos.i][aux_pos.j] == 0)
	{
		if (aux_pos.i == 7 || aux_pos.i == 0)
		{ 
			Move m(pos, aux_pos);
			m.mov_t = PROMQ;
			possible_moves.push_back(m);
			m.mov_t = PROMB;
			possible_moves.push_back(m);
			m.mov_t = PROMR;
			possible_moves.push_back(m);
			m.mov_t = PROMN;
			possible_moves.push_back(m);
		}
		else 
		{
			possible_moves.push_back(Move(pos, aux_pos));
			aux_pos.sum(-piece, 0);
			Move m(pos, aux_pos);
			m.mov_t = DOUBLE_PAWN;
			if ((pos.i == 1 || pos.i == 6) && aux_pos.is_correct() && board[aux_pos.i][aux_pos.j] == 0) {
				possible_moves.push_back(m);
			}
		}
	}
	aux_pos = pos;
	aux_pos.sum(-piece, 1);

	Position aux_en_passant(pos);
	aux_en_passant.sum(0, 1);
	if (aux_pos.is_correct() && !same_team(pos, aux_pos) && board[aux_pos.i][aux_pos.j] != 0)
	{
		if (abs(board[aux_pos.i][aux_pos.j]) == 6)
			check = true;

		if (aux_pos.i == 7 || aux_pos.i == 0)
		{
			Move m(pos, aux_pos);
			m.mov_t = PROMQ;
			possible_moves.push_back(m);
			m.mov_t = PROMB;
			possible_moves.push_back(m);
			m.mov_t = PROMR;
			possible_moves.push_back(m);
			m.mov_t = PROMN;
			possible_moves.push_back(m);
		}
		else
			possible_moves.push_back(Move(pos, aux_pos));
		
	}
	else if (last_move.mov_t == DOUBLE_PAWN && aux_en_passant.equals_pos(last_move.pfi)) {
		Move m(pos, aux_pos);
		m.mov_t = EN_PASSANT;
		possible_moves.push_back(m);
	}

	aux_pos = pos;
	aux_pos.sum(-piece, -1);
	aux_en_passant = pos;
	aux_en_passant.sum(0, -1);
	if (aux_pos.is_correct() && !same_team(pos, aux_pos) && board[aux_pos.i][aux_pos.j] != 0) 
	{
		if (abs(board[aux_pos.i][aux_pos.j]) == 6)
			check = true;

		if (aux_pos.i == 7 || aux_pos.i == 0)
		{
			Move m(pos, aux_pos);
			m.mov_t = PROMQ;
			possible_moves.push_back(m);
			m.mov_t = PROMB;
			possible_moves.push_back(m);
			m.mov_t = PROMR;
			possible_moves.push_back(m);
			m.mov_t = PROMN;
			possible_moves.push_back(m);
		}
		else
			possible_moves.push_back(Move(pos, aux_pos));
		
	}
	else if (last_move.mov_t == DOUBLE_PAWN && aux_en_passant.equals_pos(last_move.pfi)) {
		Move m(pos, aux_pos);
		m.mov_t = EN_PASSANT;
		possible_moves.push_back(m);
	}
}

void chess_board::generate_rook_moves(Position pos, std::vector<Move>& possible_moves)
{
	for (int i = -1; i < 2; i++)
		for (int j = -1; j < 2; j++)
		{
			if (abs(i) + abs(j) > 1 || (i == 0 && j == 0)) continue;
			int sum = 1;
			while (true)
			{
				Position auxP(pos);
				auxP.sum(i * sum, j * sum);

				if (!auxP.is_correct() || same_team(pos, auxP)) break;

				int piece = board[auxP.i][auxP.j];
				if (piece != 0) {
					possible_moves.push_back(Move(pos, auxP));

					if (abs(piece) == 6)
						check = true;
					break;
				}
				possible_moves.push_back(Move(pos, auxP));
				sum++;
			}
		}

}

void chess_board::generate_bishop_moves(Position pos, std::vector<Move>& possible_moves)
{
	for (int i = -1; i < 2; i+=2)
		for (int j = -1; j < 2; j+=2)
		{
			int sum = 1;
			while (true)
			{
				Position auxP(pos);
				auxP.sum(i * sum, j * sum);

				if (!auxP.is_correct() || same_team(pos, auxP)) break;

				int piece = board[auxP.i][auxP.j];
				if (piece != 0) {
					possible_moves.push_back(Move(pos, auxP));

					if (abs(piece) == 6)
						check = true;

					break;
				}
				possible_moves.push_back(Move(pos, auxP));
				sum++;
			}
		}

}

void chess_board::generate_queen_moves(Position pos, std::vector<Move>& possible_moves)
{
	for (int i = -1; i < 2; i++)
		for (int j = -1; j < 2; j++)
		{
			if ((i == 0 && j == 0)) continue;
			int sum = 1;
			while (true)
			{
				Position auxP(pos);
				auxP.sum(i * sum, j * sum);

				if (!auxP.is_correct() || same_team(pos, auxP)) break;

				int piece = board[pos.i + i * sum][pos.j + j * sum];
				if (piece != 0) {
					possible_moves.push_back(Move(pos, auxP));

					if (abs(piece) == 6)
						check = true;

					break;
				}
				possible_moves.push_back(Move(pos, auxP));
				sum++;
			}
		}

}

void chess_board::generate_knight_moves(Position pos, std::vector<Move>& possible_moves)
{
	for (int i = -1; i < 2; i += 2)
		for (int j = -1; j < 2; j += 2)
		{
			Position posAux(pos);
			posAux.sum(2 * i, j);
			if (posAux.is_correct() && !same_team(pos, posAux)) {
				int piece = board[posAux.i][posAux.j];
				
				if (abs(piece) == 6) 
					check = true;

				possible_moves.push_back(Move(pos, posAux));
			}

			posAux = pos;
			posAux.sum(i, j * 2);
			if (posAux.is_correct() && !same_team(pos, posAux)) {
				int piece = board[posAux.i][posAux.j];

				if (abs(piece) == 6)
					check = true;

				possible_moves.push_back(Move(pos, posAux));
			}
		}
}

void chess_board::generate_king_moves(Position pos, std::vector<Move>& possible_moves, bool castle_moves)
{
	for (int i = -1; i < 2; i++)
		for (int j = -1; j < 2; j++)
			if (i != 0 || j != 0) {
				Position posAux(pos);
				posAux.sum(i, j);

				if (!posAux.is_correct()) continue;
				if (!same_team(pos, posAux)) {

					possible_moves.push_back(Move(pos, posAux));

					if (abs(board[posAux.i][posAux.j]) == 6)
						check = true;
				}
				if (castle_moves && !check && abs(i) == 1 && j == 0)
				{
					if (board[pos.i][pos.j] == 6) {
						if (white_can_castle_q && no_pieces_in_between(pos, -1)) {
							posAux = pos;
							posAux.sum(0, -2);
							Move m(pos, posAux);
							m.mov_t = CASTLQ;
							possible_moves.push_back(m);
						}
						if (white_can_castle_k && no_pieces_in_between(pos, 1)) {
							posAux = pos;
							posAux.sum(0, 2);
							Move m(pos, posAux);
							m.mov_t = CASTLK;
							possible_moves.push_back(m);
						}
					}
					else {
						if (black_can_castle_q && no_pieces_in_between(pos, -1)) {
							posAux = pos;
							posAux.sum(0, -2);
							Move m(pos, posAux);
							m.mov_t = CASTLQ;
							possible_moves.push_back(m);
						}
						if (black_can_castle_k && no_pieces_in_between(pos, 1)) {
							posAux = pos;
							posAux.sum(0, 2);
							Move m(pos, posAux);
							m.mov_t = CASTLK;
							possible_moves.push_back(m);
						}
					}
				}
			}
}

bool chess_board::no_pieces_in_between(Position pos, int jump)
{
	Position auxPos(pos);
	auxPos.sum(0, jump);

	Move m(pos, auxPos);

	if (!is_legal(m)) return false;

	while (auxPos.is_correct() && board[auxPos.i][auxPos.j] == 0)
		auxPos.sum(0, jump);

	if (!auxPos.is_correct() || board[auxPos.i][auxPos.j] == 0) return true;
	if (board[pos.i][pos.j] == 6 && board[auxPos.i][auxPos.j] != 2) return false;
	if (board[pos.i][pos.j] == -6 && board[auxPos.i][auxPos.j] != -2) return false;
	return true;

}

bool chess_board::is_legal(Move& m)
{
	chess_board cb(*this);
	cb.move_piece(m);
	cb.generate_semi_legal_moves(false);
	return !cb.check;
}

void chess_board::filter_ilegal_moves(std::vector<Move>& possible_moves)
{
	int i = 0;
	for (Move& m : possible_moves) 
		if (is_legal(m)) {
			possible_moves[i] = m;
			i++;
		}
	possible_moves.resize(i);
	
	
	//std::cout << "How many ilegal?: " << possible_moves.size() - legal_moves.size() << std::endl;

}
std::vector<double> chess_board::board_to_ai_inp()
{
	const int pawns = 0;
	const int rooks = 64 * 1;
	const int knights = 64 * 2;
	const int bishops = 64 * 3;
	const int queens = 64 * 4;
	const int kings = 64 * 5;
	const int total = 64 * 6 + 1 + 4 + 1;


	std::vector<double> input(total);

	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++) {
			int piece = board[i][j];
			if (piece == 0) {
				input[pawns + i * 8 + j] = 0;
				input[rooks + i * 8 + j] = 0;
				input[knights + i * 8 + j] = 0;
				input[bishops + i * 8 + j] = 0;
				input[queens  + i * 8 + j] = 0;
				input[kings + i * 8 + j] = 0;
			}
			else if (piece == 1) input[pawns + i * 8 + j] = 1;
			else if (piece == -1) input[pawns + i * 8 + j] = -1;
			else if (piece == 3) input[knights  + i * 8 + j] = 1;
			else if (piece == -3) input[knights + i * 8 + j] = -1;
			else if (piece == 4) input[bishops  + i * 8 + j] = 1;
			else if (piece == -4) input[bishops  + i * 8 + j] = -1;
			else if (piece == 2) input[rooks + i * 8 + j] = 1;
			else if (piece == -2) input[rooks + i * 8 + j] = -1;
			else if (piece == 5) input[queens + i * 8 + j] = 1;
			else if (piece == -5) input[queens + i * 8 + j] = -1;
			else if (piece == 6) input[kings + i * 8 + j] = 1;
			else if (piece == -6) input[kings + i * 8 + j] = -1;
		}

	input[64 * 6] = white_turn;
	input[64 * 6 + 1] = black_can_castle_k;
	input[64 * 6 + 2] = black_can_castle_q;
	input[64 * 6 + 3] = white_can_castle_k;
	input[64 * 6 + 4] = white_can_castle_q;
	input[64 * 6 + 5] = last_move.mov_t == EN_PASSANT;

	return input;

}
chess_AI::chess_AI(cusparseHandle_t& handle)
{
	input = std::vector<double>(total);
	
	cusparseCreate(&handle);

	int ini_nodes = total;
	int out_nodes = 1;
	int extra_nodes = 150 + 100 + 50;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 1;
	int depth = 5;
	int delay_iteration = 0;
	double learning_rate = 0.0001;
	double error_sum = 0;
	bool adam = true;

	double inp_dropout = 0;
	std::vector<activation_function> act_f(nodes, LINEAL);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = SIGMOID;


	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = SIGMOID;

	p_rnn = Prune_evolver(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, inp_dropout, handle);
	std::vector<int> hid_layers = { 150, 100, 50 };
	p_rnn.connect_like_mlp(hid_layers, false);
	p_rnn.connect_input_output();
	p_rnn.update_rnn();

}

chess_AI::~chess_AI()
{
	
}

double chess_AI::evaluate_material(const chess_board& b, Move& m)
{
	chess_board cb(b);
	cb.move_piece(m);
	double res = 0;
	bool BKExists = false;
	bool WKExists = false;
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			if (cb.board[i][j] != 0)
			{
				if      ( cb.board[i][j] ==  1 ) res += 1;
				else if ( cb.board[i][j] == -1 ) res -= 1;
				else if ( cb.board[i][j] ==  4 
					   || cb.board[i][j] ==  3 ) res += 3;
				else if ( cb.board[i][j] == -4 
					   || cb.board[i][j] == -3 ) res -= 3;
				else if ( cb.board[i][j] ==  2 ) res += 5;
				else if ( cb.board[i][j] == -2 ) res -= 5;
				else if ( cb.board[i][j] ==  5 ) res += 9;
				else if ( cb.board[i][j] == -5 ) res -= 9;
				else if ( cb.board[i][j] ==  6 ) WKExists = true;
				else if ( cb.board[i][j] == -6 ) BKExists = true;
			}
	if (!WKExists) return -INFINITY;
	if (!BKExists) return  INFINITY;

	count++;

	return res;
}


double chess_AI::evaluate(const chess_board &b, Move& m)
{
	chess_board cb(b);
	cb.move_piece(m);

	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			if (cb.board[i][j] == 0) {
				input[pawns   * 64 + i * 8 + j] = 0;
				input[rooks   * 64 + i * 8 + j] = 0;
				input[knights * 64 + i * 8 + j] = 0;
				input[bishops * 64 + i * 8 + j] = 0;
				input[queens  * 64 + i * 8 + j] = 0;
				input[kings   * 64 + i * 8 + j] = 0;
			}
			else if (cb.board[i][j] ==  1) input[pawns   * 64 + i * 8 + j] =  1;
			else if (cb.board[i][j] == -1) input[pawns   * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] ==  3) input[knights * 64 + i * 8 + j] =  1;
			else if (cb.board[i][j] == -3) input[knights * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] ==  4) input[bishops * 64 + i * 8 + j] =  1;
			else if (cb.board[i][j] == -4) input[bishops * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] ==  2) input[rooks   * 64 + i * 8 + j] =  1;
			else if (cb.board[i][j] == -2) input[rooks   * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] ==  5) input[queens  * 64 + i * 8 + j] =  1;
			else if (cb.board[i][j] == -5) input[queens  * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] ==  6) input[kings   * 64 + i * 8 + j] =  1;
			else if (cb.board[i][j] == -6) input[kings   * 64 + i * 8 + j] = -1;


	p_rnn.forward_prop(input);
	double res = p_rnn.rnn.values_through_time[p_rnn.rnn.values_through_time.size() - 1][total]*2-1;
	p_rnn.reset_rnn();

	count++;

	return res;
}

double chess_AI::evaluate_train_material(const chess_board& b, Move& m, int white, double alfa, double beta)
{
	chess_board cb(b);
	cb.move_piece(m);

	
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			if (cb.board[i][j] == 0) {
				input[pawns + i * 8 + j] = 0;
				input[rooks  + i * 8 + j] = 0;
				input[knights  + i * 8 + j] = 0;
				input[bishops + i * 8 + j] = 0;
				input[queens + i * 8 + j] = 0;
				input[kings + i * 8 + j] = 0;
			}
			else if (cb.board[i][j] == 1) input[pawns + i * 8 + j] = 1;
			else if (cb.board[i][j] == -1) input[pawns  + i * 8 + j] = -1;
			else if (cb.board[i][j] == 3) input[knights + i * 8 + j] = 1;
			else if (cb.board[i][j] == -3) input[knights  + i * 8 + j] = -1;
			else if (cb.board[i][j] == 4) input[bishops + i * 8 + j] = 1;
			else if (cb.board[i][j] == -4) input[bishops  + i * 8 + j] = -1;
			else if (cb.board[i][j] == 2) input[rooks  + i * 8 + j] = 1;
			else if (cb.board[i][j] == -2) input[rooks  + i * 8 + j] = -1;
			else if (cb.board[i][j] == 5) input[queens + i * 8 + j] = 1;
			else if (cb.board[i][j] == -5) input[queens + i * 8 + j] = -1;
			else if (cb.board[i][j] == 6) input[kings + i * 8 + j] = 1;
			else if (cb.board[i][j] == -6) input[kings + i * 8 + j] = -1;
	
	//p_rnn->forward_prop(input);

	double mat = white * negamax(b, m, white, max_depth, alfa, beta, 0)/10;

	mat = 1 / (1 + exp(-mat));
	if (mat == INFINITY) mat = 1;
	else if(mat == -INFINITY) mat = 0;

	std::vector<double> material(1, mat);

	p_rnn.forward_prop(input);
	p_rnn.evaluate(material, false);
	p_rnn.update_weights_and_biases();
	double res = p_rnn.rnn.values_through_time[p_rnn.rnn.values_through_time.size() - 1][total]*2-1;
	p_rnn.reset_rnn();

	//std::cout << res << " == " << mat * 2 - 1 << std::endl;

	return res;
}

double chess_AI::evaluate_train_itself(const chess_board& b, Move& m, int white, double alfa, double beta)
{

	double future_evaluation = white * negamax(b, m, white, max_depth, alfa, beta, 1);

	chess_board cb(b);
	cb.move_piece(m);


	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			if (cb.board[i][j] == 0) {
				input[pawns * 64 + i * 8 + j] = 0;
				input[rooks * 64 + i * 8 + j] = 0;
				input[knights * 64 + i * 8 + j] = 0;
				input[bishops * 64 + i * 8 + j] = 0;
				input[queens * 64 + i * 8 + j] = 0;
				input[kings * 64 + i * 8 + j] = 0;
			}
			else if (cb.board[i][j] == 1) input[pawns * 64 + i * 8 + j] = 1;
			else if (cb.board[i][j] == -1) input[pawns * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] == 3) input[knights * 64 + i * 8 + j] = 1;
			else if (cb.board[i][j] == -3) input[knights * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] == 4) input[bishops * 64 + i * 8 + j] = 1;
			else if (cb.board[i][j] == -4) input[bishops * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] == 2) input[rooks * 64 + i * 8 + j] = 1;
			else if (cb.board[i][j] == -2) input[rooks * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] == 5) input[queens * 64 + i * 8 + j] = 1;
			else if (cb.board[i][j] == -5) input[queens * 64 + i * 8 + j] = -1;
			else if (cb.board[i][j] == 6) input[kings * 64 + i * 8 + j] = 1;
			else if (cb.board[i][j] == -6) input[kings * 64 + i * 8 + j] = -1;

	//p_rnn->forward_prop(input);

	future_evaluation += 1;
	future_evaluation /= 2;
	if (future_evaluation == INFINITY) future_evaluation = 1;
	else if (future_evaluation == -INFINITY) future_evaluation = 0;

	std::vector<double> future_evaluation_v(1, future_evaluation);

	p_rnn.forward_prop(input);
	p_rnn.evaluate(future_evaluation_v, false);
	double res = p_rnn.rnn.values_through_time[p_rnn.rnn.values_through_time.size() - 1][total] * 2 - 1;
	p_rnn.reset_rnn();

	//std::cout << res << " == " << mat * 2 - 1 << std::endl;

	return res;
}

double chess_AI::negamax(const chess_board& b, Move m, int white, int depth, double alfa, double beta, int eval)
{
	if (depth <= 0 ) {
		if (eval == 0)      return white * evaluate_material(b, m);
		else if (eval == 1) return white * evaluate(b, m);
		else if (eval == 2) return white * evaluate_train_material(b, m, white, alfa, beta);
		else if (eval == 3) return white * evaluate_train_itself(b, m, white, alfa, beta);
	}

	chess_board cb(b);
	cb.move_piece(m);
	if (cb.fifty_moves_counter <= 0) return 0;

	std::vector<Move> moves = cb.generate_semi_legal_moves(true);
	cb.filter_ilegal_moves(moves);

	double bestScore = -INFINITY;
	for (Move& mi : moves)
	{
		//if (max_depth > depth && float(rand() % 1000) / 1000 < 0.75 )
		//	continue;

		//std::cout << max_depth << " " << depth << " " << float((max_depth - depth)) / max_depth << std::endl;
		double score = -negamax(cb, mi, -white, depth - 1, -beta, -alfa, eval);

		alfa = std::max(alfa, score);

		bestScore  = std::max(score, bestScore);

		if (alfa > beta)
			return beta;

	}

	//if(depth == 3)std::cout << worstScore << std::endl;
	if (moves.size() == 0) {
		cb.white_turn = !cb.white_turn;
		cb.generate_semi_legal_moves(false);

		if (!cb.check) {

			//std::cout << "STALEMATE!" << std::endl;
			return 0;
		}
	}

	count++;

	return bestScore;
}

void chess_AI::play_best_move(chess_board& b, int depth, int eval)
{
	std::vector<Move> best_moves;

	std::vector<Move> moves = b.generate_semi_legal_moves(true);
	b.filter_ilegal_moves(moves);
	 
	double score = -INFINITY;
	int whiteInt = 1;
	if (!b.white_turn) whiteInt = -1;


	//println("-----------------");
	for (Move& mi : moves)
	{
		double aux_score = -negamax(b, mi, -whiteInt, depth, -INFINITY, INFINITY, eval);

		//std::cout << aux_score << std::endl;
		if (aux_score > score)
		{
			best_moves.clear();
			best_moves.push_back(mi);
			score = aux_score;
		}
		else if (aux_score == score)
			best_moves.push_back(mi);

	}
	std::cout<<"Best score: " << score << std::endl;
	b.move_piece(best_moves[rand() % best_moves.size()]);

}

void chess_2AI_problem()
{

}
void test_fen()
{
	chess_board cb("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
	cb.print_board();
	cb = chess_board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
	cb.print_board();
}
void test_board()
{

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	chess_AI cai(handle);
	chess_board cb;
	cb.print_board();

	int n;
	//srand(time(NULL));
	int partides = 100;
	int max_weights = 100000;
	int connections_to_prune = 0;
	cai.p_rnn.rnn.bias_learning_rate = 0;
	for (int i = 0; i < partides; i++)
	{

		cb = chess_board();
		std::cout << "Partida " << i << std::endl;
		int i_move = 1000;
        
		while (true)
		{
			cb.print_board();
			std::vector<Move> moves = cb.generate_semi_legal_moves(true);
			cb.filter_ilegal_moves(moves);
			if (moves.size() == 0)
				break;

			//Move m = moves[rand() % moves.size()];
			if (cb.white_turn)
				std::cout << "White turn" << std::endl;
			else
				std::cout << "Black turn" << std::endl;
			std::cout << "Moves to draw: " << cb.fifty_moves_counter << std::endl;
			
			//std::cout << m.pini.i << " " << m.pini.j << ",  " << m.pfi.i << " " << m.pfi.j << std::endl;
			//cb.move_piece(m);
			if (cb.white_turn) cai.play_best_move(cb, 1, 2);
			else cai.play_best_move(cb, 1, 2);

			std::cout <<"Count: " << cai.count << std::endl;
			cai.count = 0;

			cai.p_rnn.update_weights_and_biases();
			/*
			if (i_move > 100)
			{
				cai.p_rnn.add_n_random_connections(cai.p_rnn.rnn.weights.nnz * 0.2);
				connections_to_prune = max_weights - cai.p_rnn.rnn.weights.nnz;
				i_move = 0;
			}
			else
			{
				cai.p_rnn.read_weights_and_biases_from_device();

				cai.p_rnn.prune_weights(connections_to_prune/100);
				cai.p_rnn.update_rnn();
			}*/
			i_move++;

		}

		cb.white_turn = !cb.white_turn;
		std::vector<Move> moves = cb.generate_semi_legal_moves(true);

		if (cb.check && cb.fifty_moves_counter > 0) {
			if (cb.white_turn)
				std::cout << "White wins!" << std::endl;
			else
				std::cout << "Black wins!" << std::endl;
		}
		else
			std::cout << "Draw" << std::endl;
		cb.print_board();

	}
	/*
	cb.print_board();
	int i1, j1;
	while (std::cin >> i1 >> j1)
	{
		std::vector<Move> moves = cb.generate_semi_legal_moves(true);
		cb.filter_ilegal_moves(moves);
		std::cout << moves.size() << std::endl;
		int i = -1;

		for (int it = 0; it < moves.size(); it++)
			if (i1 == moves[it].pini.i && j1 == moves[it].pini.j )
				std::cout <<"Move "<<it<<": "<< moves[it].pfi.i << " " << moves[it].pfi.j <<", " << cb.board[moves[it].pfi.i][moves[it].pfi.j]<< std::endl;
		
		std::cin >> i;

		if (i > moves.size() || i < 0)
			break;

		cb.move_piece(moves[i]);
		cb.print_board();
	}*/
	/*Position pini(1, 4);
	Position pfi(4, 4);
	Move m(pini, pfi);
	cb.move_piece(m);
	cb.print_board();
	pini = Position(6, 5);
	pfi = Position(4, 5);
	m = Move(pini, pfi);
	cb.move_piece(m);
	cb.print_board();

	pini = Position(4, 4);
	pfi = Position(5, 5);
	m = Move(pini, pfi);
	m.mov_t = EN_PASSANT;

	cb.move_piece(m);
	cb.print_board();*/
}