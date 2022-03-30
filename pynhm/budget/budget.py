import warnings

# from ..base.storageUnit import StorageUnit
from ..utils.dictionary_as_properties import DictionaryAsProperties

# JLM: maybe this is a base class?
# JLM: seems like every storage unit should have a budget

# IDEA: Could require dict entires to be dicts themselves with "data" and "component"
# entries for managing budget sub component. Do that after a basic budget
# is working with out subcomponents/categories
# OR: just subclass this budget to work with different parts of the model:
# e.g. ET, runoff, GW. Some of these might have different spatial dimensions (
# like run off getting mapped to streamflow vs elsewhere).

# IDEA: Maintain a timeseries? or not? not at first?

# Right now subclassing from StorageUnit is counterproductive.
#   * budget does not have parameters, does a storage unit? nor methods on these.
#   * budget does not have an atm, does a storage unit?
#   * what is id? (doc strings)
#   * what is storage type? (doc strings0


class Budget:
    def __init__(self, data: dict = None, verbosity: int = 0):

        self.verbosity = verbosity
        self._budget = {}
        self._terms = {}
        # JLM: not sure why i'd use DictionaryAsProperties({})

        if data is not None:
            for name, obj in data.items():
                self.add_term(name, obj)

        return None

    def add_term(self, name, obj):
        # JLM do we want to support renaming from obj to budget or not?
        self._terms[name] = obj
        return None

    # with categories eventually?
    @property
    def terms(self):
        return list(self._terms.keys())

    @property
    def budget(self):
        return self._budget

    @property
    def balance(self):
        return self._budget["balance"]

    def calculate(self):
        terms = self.terms
        if len(terms) <= 1:
            warnings.warn(
                f"Budget has {len(terms)} term(s), nothing to calculate."
            )
        ref_time = self._terms[terms[0]].current_time
        for tt in terms:
            assert ref_time == self._terms[tt].current_time
            # JLM: renaming would allow 2 left tts to differ from tt on right
            self._budget[tt] = self._terms[tt].get_current_state(tt)
            if tt == terms[0]:
                self._budget["balance"] = self._budget[tt]
            else:
                self._budget["balance"] += self._budget[tt]
