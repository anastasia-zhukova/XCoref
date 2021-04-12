"""
A one to n set.

Has all similar abilities to normal lists.
The limitation of one-to-n-sets is that each item can only be contained once in a set
and each item can only be associated to one set (the relation is set 1:n item).

The list functions are slightly altered in most cases.
The set ensures that an item will stay in it but does not ensure that it will always have the same index:

.. code-block:: python

    >>> set_1 = OneToNSet([item1, item2, item3])  # Parents of all items will be set
    >>> set_2 = OneToNSet()
    >>> set_2[0] = item1   # item1 will be removed from set_1 and added to set_2
                           # -> Index item2 and item3 will be lowered

.. note::
    The indices won't change if only one list is used and items are not moved from one to another list.

The slicing notation returns a list instead of a OneToNSets:

.. code-block:: python

    >>> test_set = OneToNSet([item1, item2, item3])
    >>> x = test_set[:1]
    >>> type(x)
    <class 'list'>
    >>> x = test_set + [doc4]
    >>> type(x)
    <class 'list'>
    >>> test_set += [doc5]
    >>> type(test_set)
    <class 'structures.one_to_n_set.OneToNSet'>
    >>> doc_set *= 3
    TypeError: unsupported operand type(s) for *=: '<class 'structures.one_to_n_set.OneToNSet'>' and '<class 'int'>'

.. note::
    It will always be ensured, that the childs parent always contains the child and that the childs parent is always
    only contained in that one OneToNSet.
"""
from typing import Optional, Iterable, Any
from contextlib import suppress
NOT_PASSED = object()


class OneToNSet(list):
    CHILDRENS_PARENT_ATTRIBUTE = "parent"
    """
    str: Name of the attribute which contains the parent item in the child item.
    E.g. if ``CHILDRENS_PARENT_ATTRIBUTE = "xyz"`` the childs parent attribute is ``child.xyz``.
    """

    CHILDRENS_INDEX_ATTRIBUTE = None
    """
    str, `optional`: Name of the attribute to assign its index to. Won't add any index if `None`.
    The index will be updated on change. 
    """

    def __init__(self, items: Iterable = [], force: bool = False):
        """
        Create an one-to-n set.

        Args:
            items: Iterable of items to add to the set.
            force: Whether the items should be removed from the other OneToNSet if contained in another.
                Throws an AssertionError if ``False`` and any item is contained in another set.
        """
        super().__init__()
        self.extend(items, force)

    def append(self, item: Any, force: bool = False) -> Any:
        """
        Add an item to the end of the list.

        Args:
            item: Item to add.
            force: Whether the item should be removed from the other OneToNSet if contained in another.
                Throws an AssertionError if ``False`` and contained in another set.

        Returns:
            The item (parsed if a parser was used in a class inheriting from this).
        """
        item, old_parent = self._test_and_parse(item, force)
        try:
            super().append(item)
        except Exception:
            # Revert changes if it cannot be appended and raise error again.
            self.__set_childs_parent(item, old_parent)
            raise
        return item

    def extend(self, items: Any, force: bool = False) -> Any:
        """
        Extend the list by appending all items from the iterable.

        Args:
            items: Iterable of items to append.
            force: Whether the items should be removed from the other OneToNSet if contained in another.
                Throws an AssertionError if ``False`` and any item is contained in another set.

        Returns:
            The items (parsed if a parser was used in a class inheriting from this).
        """
        checked = []
        for item in items:
            checked.append(self._test_and_parse(item, force))
        checked_items = list(map(lambda x: x[0], checked))
        try:
            super().extend(checked_items)
        except Exception:
            # Revert changes if it cannot be appended and raise error again.
            with suppress(ValueError):
                for item, old_parent in checked:
                    self.remove(item)
                    self.__set_childs_parent(item, old_parent)
            raise
        return checked_items

    def insert(self, i: int, item: Any, force: bool = False) -> Any:
        """
        Insert an item at a given position.

        Args:
            i: Index of the element before which to insert, so ``a.insert(0, item)`` inserts at the front of the
                list, and ``a.insert(len(a), x)`` is equivalent to ``a.append(x)``.
            item: item to add.
            force: Whether the item should be removed from the other OneToNSet if contained in another.
                Throws an AssertionError if ``False`` and contained in another set.

        Returns:
            The item (parsed if a parser was used in a class inheriting from this).
        """
        item, old_parent = self._test_and_parse(item, force)
        try:
            super().insert(i, item)
        except Exception as e:
            # Revert changes if it cannot be appended and raise error again.
            self.__set_childs_parent(item, old_parent)
            raise e
        return item

    def remove(self, item: Any) -> None:
        """
        Remove the first item from the list whose value is equal to item. It raises a ValueError if there is no
        such item.

        Args:
            item: item to remove from the list.
        """
        super().remove(item)
        self.__set_childs_parent(item, None)

    def pop(self, i: Optional[int] = None) -> Any:
        """
        Remove the item at the given position in the list, and return it.
        If no index is specified, ``a.pop()`` removes and returns the last item in the list.

        Args:
            i: The position of the item to remove. Last if `None`.

        Returns:
            The removed item.
        """
        if i is None:
            item = super().pop()
        else:
            item = super().pop(i)
        self.__set_childs_parent(item, None)
        return item

    def clear(self) -> None:
        """
        Remove all items from the list.
        """
        while len(self):
            self.pop()

    def _test_and_parse(
            self,
            item: Any,
            force: bool = False,
            parent_set: bool = False,
            old_parent: Any = NOT_PASSED
    ) -> Any:
        """
        Test if an item is correct to insert into the set.

        Checks if the item is not contained in another OneToNSet.
        (Parses the item if a parser was used in a class inheriting from this).

        Updates the parent

        Args:
            item: Item being added (or already manually added).
            force: Whether the item should be removed from the other OneToNSet if contained in another.
                Throws an AssertionError if ``False`` and contained in another set.
            parent_set: Whether the parent was already set beforehand. Requires `old_parent` if `True`.
            old_parent: Parent the item had before it was already set beforehand. (The parent to reset
                to if something goes wrong.)

        Returns:
            The item (parsed if a parser was used in a class inheriting from this).
        """
        if parent_set:
            if old_parent is NOT_PASSED:  # None is acutally allowed, thus use NOT_PASSED
                raise ValueError("Parent was already set before but old_parent is not passed.")
        else:
            old_parent = self.__get_childs_parent(item)

          # old_parent is not self is required for using deepcopy.
        if not force and old_parent and parent_set is False and old_parent is not self:
            # Item already is already contained in another set.
            raise AssertionError("Item can not be inserted into the OneToNSet, as it is already contained in another "
                                 "one. Set force to true to remove it from the other list and add it to this list.")

        if item in self:
            if parent_set:
                self.__set_childs_parent(item, old_parent)
            raise ValueError("Item is already contained in the OneToNSet.")

        if not parent_set:
            self.__update_parent(item)

        return item, old_parent

    def __update_parent(self, child: Any) -> None:
        """
        Update the parent of an item. Removes it from the old set if necessary.

        Args:
            item: item to update the parent from.
        """
        childs_parent = self.__get_childs_parent(child)

        if childs_parent and childs_parent is not self:  # childs_parent is not self is required for using deepcopy.
            childs_parent.remove(child)

        self.__set_childs_parent(child, self)

    def __set_childs_parent(self, child: Any, parent: Any) -> None:
        """
        Set the parent of a child to parent.

        Args:
            child: Child to set the parent from.
            parent: Parent to set. (Normally ``self`` or ``None``.)
        """
        setattr(child, self.CHILDRENS_PARENT_ATTRIBUTE, parent)

    def __get_childs_parent(self, child: Any) -> Any:
        """
        Get the parent of a child.

        Args:
            child: Child to get the parent from.

        Returns:
            Parent of the child.
        """
        return getattr(child, self.CHILDRENS_PARENT_ATTRIBUTE, None)

    def __delitem__(self, key):
        # Set parent to none for the deleted item.
        child = self[key]
        super().__delitem__(key)
        self.__set_childs_parent(child, None)

    def __str__(self):
        # Add class name before brackets. E.g. OneToNSet[].
        cls = self.__class__
        return cls.__name__ + super().__repr__()

    def __repr__(self):
        # Add the module and name before the brackets. E.g. structures.one_to_n_set.OneToNSet[].
        cls = self.__class__
        return cls.__module__ + "." + cls.__name__ + super().__repr__()

    def __iadd__(self, items) -> None:
        # Extending via ``one_to_n_set += [doc4, doc5]``.
        self.extend(items)
        return self

    def __imul__(self, other):
        # Disable multiplying the class (*=).
        type_self = type(self)
        type_other = type(other)
        raise TypeError(f"unsupported operand type(s) for *=: '{type_self}' and '{type_other}'")

    def __setitem__(self, key, item):
        # Update parent of item when assigned via ``on_to_n_set[0] = item``
        super().__setitem__(key, item)
        self.__update_parent(item)

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__
