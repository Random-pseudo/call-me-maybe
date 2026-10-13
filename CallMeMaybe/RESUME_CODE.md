# Résumé du Code — Version ELI15 (Explique-moi comme si j'avais 15 ans)

> **L'idée générale :** Un petit modèle d'IA (LLM de 0.6B) est comme un perroquet ultra-intelligent mais très bavard et distrait. Si on lui demande gentiment du JSON, il invente du faux texte, oublie des virgules ou se trompe dans les types.  
> Notre code agit comme un **tuteur et des rails de sécurité** : à chaque mot ou symbole, on interdit physiquement à l'IA de sortir de la route pour obtenir un JSON 100 % parfait qui appelle la bonne fonction.

---

## 1. Modèles de Données (`src/models/`) : Le Contrôle de Sécurité (Pydantic)

Ces classes vérifient que les données qui entrent et sortent respectent scrupuleusement les règles du jeu.

### `src/models/functions.py`
- **`class ParameterSchema(BaseModel)`** : La fiche d'identité d'un paramètre (ex. : *"Est-ce un nombre ? Du texte ? Y a-t-il une liste de choix imposés ?"*).
- **`class ReturnSchema(BaseModel)`** : L'étiquette qui indique le type de résultat retourné par la fonction (un nombre, une phrase, etc.).
- **`class FunctionDefinition(BaseModel)`** : La fiche technique complète d'un outil (son nom, ce qu'il sait faire, ses paramètres et son résultat).
- **`class FunctionsDefinitionList(RootModel[List[FunctionDefinition]])`** : La boîte à outils complète : valide que la liste de toutes nos fonctions dans le fichier JSON est propre.

### `src/models/input_data.py`
- **`class PromptItem(BaseModel)`** : Une question posée par l'utilisateur (ex. : *"Combien font 2 plus 3 ?"*).
- **`class QueryList(RootModel[List[PromptItem]])`** : Le carnet de commandes contenant la liste de toutes les questions utilisateurs à traiter.

### `src/models/output_data.py`
- **`class FunctionCallResult(BaseModel)`** : Le formulaire officiel de réponse avec strictement 3 cases obligatoires (`prompt`, `fn_name`, `args`) et interdiction absolue d'ajouter quoi que ce soit d'autre.
- **`class FunctionCallResultList(RootModel[List[FunctionCallResult]])`** : Le classeur final regroupant toutes les réponses propres avant de les sauvegarder dans le fichier JSON de sortie.

---

## 2. Décodage Contraint (`src/constrained_decoding/`) : Les Garde-Fous

C'est ici qu'on force l'IA à rester dans les clous token par token.

### `src/constrained_decoding/mask_generator.py`
- **`def apply_logit_mask(logits, allowed_token_indices, mask_value=-np.inf)`** : Le videur de boîte de nuit : il colle une note impossible ($-\infty$) à tous les mots interdits pour que l'IA ne puisse choisir qu'entre les mots autorisés.

### `src/constrained_decoding/json_schema_parser.py`
- **`class ParsedParameterSchema`** : Une version ultra-simple et rangée d'un paramètre, facile et rapide à lire pour notre programme.
- **`class ParsedFunctionSchema`** : La notice d'un outil prête à l'emploi que le générateur consulte pour savoir ce qui est permis.
- **`def parse_function_definitions(definitions)`** : Le traducteur qui prend les définitions brutes et fabrique les notices faciles à manipuler par le code.

### `src/constrained_decoding/fsa_state_machine.py`
- **`class States(str, Enum)`** : Les 15 cases d'un jeu de l'oie (Début, écrire le nom de la fonction, ouvrir les arguments, écrire la valeur, fermer les accolades, Terminé).
- **`class FSAJSON`** : L'arbitre du jeu (l'automate fini) qui surveille l'IA à chaque lettre générée pour s'assurer qu'elle respecte la syntaxe JSON.
  - **`__init__(self, schemas, model)`** : Prépare l'arbitre avec les règles et apprend par cœur les symboles clés (`{`, `}`, `,`, `:`).
  - **`_encode_first(self, text)`** : Traduit un symbole en code numérique unique compris par l'IA.
  - **`is_finished(self)`** : Vérifie si la recette est finie et si le JSON est bien refermé (case `DONE`).
  - **`get_valid_token_ids(self)`** : Donne à l'IA la liste des seuls mots ou symboles qu'elle a le droit de dire à l'instant T.
  - **`_tokens_for_target(self, target)`** : Souffle à l'IA les lettres suivantes nécessaires pour épeler exactement le mot attendu.
  - **`_get_allowed_param_value_tokens(self)`** : Sélectionne les seuls caractères autorisés pour une valeur (ex. que des chiffres pour un nombre, que "true"/"false" pour un booléen).
  - **`step(self, token_id)`** : Fait avancer le jeu d'une case dès que l'IA a posé un symbole valide.
  - **`_prepare_next_param(self)`** : Passe au paramètre suivant sur la liste une fois le précédent terminé.
  - **`_handle_param_value_transition(self)`** : Récupère la valeur saisie par l'IA (chiffre, texte) et l'enregistre quand elle est complète.
  - **`_advance_after_param(self)`** : Décide s'il faut poser une virgule pour un autre argument ou refermer l'accolade si c'était le dernier.
  - **`get_parsed_json(self)`** : Récupère le texte complet généré et le transforme en un vrai dictionnaire Python propre.

---

## 3. Pipeline d'Inférence (`src/pipeline/`) : Le Chef d'Orchestre

C'est ici qu'on fait réfléchir l'IA pour choisir la bonne fonction et remplir ses paramètres.

### `src/pipeline/function_selector.py`
- **`def format_functions_doc(schemas)`** : Rédige un menu clair décrivant tous les outils disponibles pour le montrer à l'IA.
- **`def select_function_with_llm(model, query, schemas, system_prompt)`** : Fait voter l'IA pour l'outil le plus logique en calculant mathématiquement sa probabilité (aucune triche ni devinette manuelle).

### `src/pipeline/generator.py`
- **`def generate_parameter_value(model, input_ids, param_name, param_schema, is_last)`** : Force l'IA à écrire la valeur d'un paramètre (un chiffre, un mot, un choix) en bloquant tout caractère hors-sujet.
- **`def generate_function_call(model, query, schemas, system_prompt)`** : Gère une question de bout en bout : fait choisir la fonction à l'IA puis remplit tous ses paramètres un par un.
- **`def run_pipeline(functions_path, input_path, output_path, model)`** : La chaîne de production : ouvre les fichiers, traite toutes les questions avec l'IA et enregistre le résultat dans le fichier JSON final.

---

## 4. Point d'Entrée (`src/__main__.py`) : La Porte d'Entrée

- **`def parse_arguments()`** : Lit ce que l'utilisateur a tapé dans son terminal (options `--input`, `--output`, etc.).
- **`def resolve_functions_path(explicit_path)`** : Trouve tout seul le bon fichier d'outils même si l'utilisateur utilise un nom légèrement différent.
- **`def main()`** : Le gardien : vérifie en moins de 0.05 seconde que les fichiers existent et sont sans faute de frappe avant d'allumer l'IA, évitant tout crash inutile.

---

## 5. SDK du Modèle (`llm_sdk/small_llm_model.py`) : Le Cerveau de l'IA (Fourni)

- **`class Small_LLM_Model`** : La télécommande pour charger et faire tourner le modèle de langage Qwen3-0.6B en local.
  - **`__init__(self, model_name, device, dtype, trust_remote_code)`** : Allume le modèle et charge son dictionnaire de mots dans la mémoire de l'ordinateur.
  - **`encode(self, text)`** : Traduit une phrase humaine en une liste de numéros (tokens) compréhensibles par la machine.
  - **`decode(self, token_ids)`** : Fait l'inverse : traduit une suite de numéros en phrase lisible par un être humain.
  - **`get_logits_from_input_ids(self, input_ids)`** : Demande à l'IA de donner son score de confiance (logit) pour chaque mot possible du dictionnaire pour la suite.
  - **`get_path_to_vocabulary_json(self)`** : Donne l'adresse du dictionnaire qui relie chaque mot à son numéro de code.

---

## 6. Tests Automatisés (`tests/test_constrained_decoding.py`) : Le Crash-Test

- **`class TestConstrainedDecoding(unittest.TestCase)`** : Le banc d'essai qui vérifie automatiquement que tout notre système de sécurité fonctionne.
  - **`test_apply_logit_mask_basic(self)`** : Vérifie que le videur élimine bien tous les mots interdits et ne laisse passer que les bons.
  - **`test_apply_logit_mask_empty_raises(self)`** : Vérifie que le système tire la sonnette d'alarme si on lui donne une liste vide de mots autorisés.
  - **`test_parse_function_definitions(self)`** : Vérifie que les définitions d'outils sont bien traduites sans rien perdre.
  - **`test_function_call_result_valid(self)`** : Vérifie qu'un résultat bien formé est accepté avec succès.
  - **`test_function_call_result_extra_forbidden(self)`** : Vérifie qu'on se fait bien recaler si on tente d'ajouter un champ imprévu dans la sortie.
  - **`test_fsa_state_transitions(self)`** : Vérifie que l'automate avance correctement sur le jeu de l'oie sans sauter de case.
